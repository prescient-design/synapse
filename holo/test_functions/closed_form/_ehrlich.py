import torch
from botorch.test_functions import SyntheticTestFunction
import pickle
import numpy as np
import pandas as pd
import random
from constants._amino_acids import aa_to_idx

from holo.test_functions.elemental import (
    dmp_sample_log_likelihood,
    dmp_stationary_dist,
    motif_search,
    sample_dmp,
    sample_sparse_ergodic_transition_matrix,
)

# Lazy loading cache for the parquet data to avoid OOM at import time
_oas_df_cache = None

def _get_oas_dataframe():
    """Lazily load and cache the OAS paired data."""
    global _oas_df_cache
    if _oas_df_cache is None:
        df = pd.read_csv("sequence_examples.csv")
        df = df[df['fv_heavy_aho'].apply(len) == 149]
        df = df[df['fv_light_aho'].apply(len) == 148]
        _oas_df_cache = df.reset_index(drop=True)
    return _oas_df_cache

def clear_oas_cache():
    """Clear the OAS dataframe cache to free memory after Ehrlich functions are created."""
    global _oas_df_cache
    if _oas_df_cache is not None:
        del _oas_df_cache
        _oas_df_cache = None
        import gc
        gc.collect()


class Ehrlich(SyntheticTestFunction):
    _optimal_value = 1.0
    num_objectives = 1

    def __init__(
        self,
        num_states: int = 5,
        dim: int = 7,
        num_motifs: int = 1,
        motif_length: int = 3,
        quantization: int | None = None,
        epistasis_factor: float = 0.0,
        noise_std: float = 0.0,
        negate: bool = False,
        random_seed: int = 0,
    ) -> None:
        bounds = [(0.0, float(num_states - 1)) for _ in range(dim)]
        self.num_states = num_states
        self.dim = dim
        self.continuous_inds = []
        self.discrete_inds = list(range(self.dim))
        self.categorical_inds = []
        self._random_seed = random_seed
        self._motif_length = motif_length
        self._quantization = quantization

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
        )
        self._generator = torch.Generator().manual_seed(random_seed)
        self._epistasis_factor = epistasis_factor
        self.initial_dist = torch.ones(num_states) / num_states
        bandwidth = int(num_states * 0.4)

        # sample an initial sequence from paired data based on torch generator
        df = _get_oas_dataframe()
        random_idx = torch.randint(0, len(df), (1,), generator=self._generator).item()
        self.initial_sequence = df.iloc[random_idx]['fv_heavy_aho'] + df.iloc[random_idx]['fv_light_aho']
        self.initial_sequence_torch = torch.tensor([aa_to_idx[char] for char in self.initial_sequence])



        self.motifs, self.spacings = generate_spaced_motifs(self.initial_sequence_torch, num_motifs, motif_length, self._generator)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        motif_contrib = []
        for motif, spacing in zip(self.motifs, self.spacings):
            motif_present = motif_search(
                solution=X,
                motif=motif,
                spacing=spacing,
                mode="count",
                quantization=self._quantization,
            )
            response = _cubic_response(motif_present, self._epistasis_factor)
            motif_contrib.append(response) # scale up the response by 100

        all_motifs_contrib = torch.stack(motif_contrib).sum(dim=0) # changed this from prod
       
        return all_motifs_contrib

    def _validate_inputs(self, X: torch.Tensor) -> None:
        """Custom input validation to work around BoTorch 0.15.1 validate_inputs bug."""
        if not X.shape[-1] == self.dim:
            raise ValueError(
                "Expected `X` to have shape `(batch_shape) x n x d`. "
                f"Got {X.shape=} and {self.dim=}"
            )
        if not ((X >= self.bounds[0]).all() and (X <= self.bounds[1]).all()):
            raise ValueError(
                f"Input values must be within bounds {self.bounds}. "
                f"Got min: {X.min()}, max: {X.max()}"
            )
        # Check discrete parameters are integer-valued
        if self.discrete_inds:
            discrete_X = X[..., self.discrete_inds]
            if not torch.allclose(discrete_X, discrete_X.round()):
                raise ValueError("Discrete parameters must be integer-valued")

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Override to work around BoTorch 0.15.1 validate_inputs bug."""
        #self._validate_inputs(X)
        return self._evaluate_true(X=X)

    def evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        """Override to work around BoTorch 0.15.1 validate_inputs bug."""
        self._validate_inputs(X)
        # Call parent implementation if it exists, otherwise return empty tensor
        try:
            return super()._evaluate_slack_true(X=X)
        except AttributeError:
            # No constraint slacks for this function
            return torch.empty(X.shape[:-1] + (0,), dtype=X.dtype, device=X.device)


    def to(self, device, dtype):
        self.initial_sequence = self.initial_sequence.to(device)
        self.motifs = [motif.to(device) for motif in self.motifs]
        self.spacings = [spacing.to(device) for spacing in self.spacings]
        self._generator = torch.Generator(device=device).manual_seed(self._random_seed)
        return self

    def __repr__(self):
        motif_list = [f"motif_{i}: {motif.tolist()}" for i, motif in enumerate(self.motifs)]
        spacing_list = [f"spacing_{i}: {spacing.tolist()}" for i, spacing in enumerate(self.spacings)]
        return (
            f"Ehrlich("
            f"num_states={self.num_states}, "
            f"dim={self.dim}, "
            f"num_motifs={len(self.motifs)}, "
            f"motifs=[{', '.join(motif_list)}], "
            f"spacings=[{', '.join(spacing_list)}], "
            f"quantization={self._quantization}, "
            f"noise_std={self.noise_std}, "
            f"negate={self.negate}, "
            f"random_seed={self._random_seed})"
        )


def _cubic_response(X: torch.Tensor, epistasis_factor: float):
    coeff = epistasis_factor * X * (X - 1.0) + 1.0
    return coeff * X


def generate_spaced_motifs(sequence, num_motifs, motif_length, generator, ignore_token=0, spacing_range=(1, 2)):
    """
    Generate motifs with adaptive spacing that fits within the available sequence.
    
    Args:
        sequence: 1D tensor representing the sequence
        num_motifs: Number of motifs to generate
        motif_length: Length of each motif
        ignore_token: Token to ignore (default 0)
        spacing_range: Tuple of (min_spacing, max_spacing)
    
    Returns:
        Tuple of (motifs, spacings)
    """
    # Filter out ignore_token. Can make this the 0 token in the future if we want to ignore gaps
    non_ignore_mask = sequence != ignore_token
    non_ignore_sequence = sequence[non_ignore_mask]
    
    if len(non_ignore_sequence) < motif_length:
        raise ValueError(f"Sequence too short. Need at least {motif_length} non-zero tokens, got {len(non_ignore_sequence)}")
    
    motifs = []
    spacings = []
    min_spacing, max_spacing = spacing_range
    
    for motif_idx in range(num_motifs):
        # Try different starting positions until we find one that works
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            # Choose a random starting position
            start_idx = random.randint(0, len(non_ignore_sequence) - motif_length)
            
            # Calculate available space for this starting position
            remaining_length = len(non_ignore_sequence) - start_idx
            max_total_spacing = remaining_length - motif_length
            
            if max_total_spacing >= (motif_length - 1) * min_spacing:
                # We can fit a motif here, generate spacings
                motif_spacings = []
                remaining_spacing_budget = max_total_spacing
                
                for i in range(motif_length - 1):
                    if i == motif_length - 2:  # Last spacing
                        # Use all remaining budget or max_spacing, whichever is smaller
                        spacing = min(remaining_spacing_budget, max_spacing)
                    else:
                        # Leave room for remaining elements
                        remaining_elements = motif_length - 2 - i
                        max_this_spacing = min(
                            max_spacing, 
                            remaining_spacing_budget - remaining_elements * min_spacing
                        )
                        spacing = torch.randint(min_spacing, max(min_spacing, max_this_spacing) + 1, (1,), generator=generator).item()
                    
                    motif_spacings.append(spacing)
                    remaining_spacing_budget -= spacing
                
                # Extract the motif elements
                motif_elements = []
                current_pos = start_idx
                
                for i in range(motif_length):
                    motif_elements.append(non_ignore_sequence[current_pos])
                    if i < motif_length - 1:
                        current_pos += motif_spacings[i] + 1
                
                motif = torch.tensor(motif_elements)
                motifs.append(motif)
                spacings.append(torch.tensor(motif_spacings))
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not generate motif {motif_idx} with given constraints")
    
    print(f"Generated motifs: {motifs}")
    print(f"Generated spacings: {spacings}")
    return tuple(motifs), spacings
