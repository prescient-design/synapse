"""Synthetic dataset generation using Ehrlich test functions."""

import gc
import random
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from holo.test_functions.closed_form import Ehrlich
from constants._connectivity import CONNECTIVITY_TYPES
from constants._amino_acids import aa_to_idx
from utils import complex_format_fn

_position_probs_cache = {}


def _get_position_probabilities(chain_type):
    if chain_type not in _position_probs_cache:
        path = f'synthetic_dataset/oas_position_probabilities_{chain_type}.pkl'
        with open(path, 'rb') as f:
            _position_probs_cache[chain_type] = pickle.load(f)
    return _position_probs_cache[chain_type]


def sequence_sampler(initial_sequence, edit_prob, num_samples):
    heavy_probs = _get_position_probabilities('heavy')
    light_probs = _get_position_probabilities('light')
    all_aa = list(aa_to_idx.keys())

    sequences = []
    for _ in range(num_samples):
        seq = []
        for idx, char in enumerate(initial_sequence):
            if char != '-' and random.random() < edit_prob:
                probs = heavy_probs[idx] if idx < 149 else light_probs[idx - 149]
                p = [probs.get(aa, 0.0) for aa in all_aa]
                total = sum(p)
                p = [x / total for x in p] if total > 0 else [1.0 / len(all_aa)] * len(all_aa)
                seq.append(aa_to_idx[np.random.choice(all_aa, p=p)])
            else:
                seq.append(aa_to_idx[char])
        sequences.append(seq)
    return sequences


class AntibodySyntheticDataset(Dataset):
    def __init__(self, num_states, num_motifs, initial_seed, num_samples, noise_level=0.1,
                 edit_prob=0.1, input_type='trispecific', connectivity_filter=None,
                 ehrlich_functions=None, epistasis_factor=0.0, motif_length=3):

        self.input_type = input_type
        fabs_map = {'monospecific': 1, 'bispecific': 2, 'trispecific': 3, 'trispecific_example': 3,
                    'tetraspecific': 4, 'pentaspecific': 5}
        self.num_fabs = fabs_map.get(input_type)
        if self.num_fabs is None:
            raise ValueError(f"Invalid input_type: {input_type}")

        self.num_states = num_states
        self.num_motifs = num_motifs
        self.motif_length = motif_length
        self.random_seed = initial_seed
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.edit_prob = edit_prob
        self.connectivity_filter = connectivity_filter
        self.epistasis_factor = epistasis_factor
        self.ehrlich_functions = ehrlich_functions

        self.data = self._preprocess_data()

    def _preprocess_data(self):
        # Generate or use provided Ehrlich functions
        if self.ehrlich_functions is None:
            self.ehrlich_functions = [
                Ehrlich(num_states=self.num_states, num_motifs=self.num_motifs,
                       motif_length=self.motif_length, dim=297,
                       random_seed=i + self.random_seed, epistasis_factor=self.epistasis_factor)
                for i in range(self.num_fabs)
            ]

        # Generate sequences and function values for each fab
        fab_sequences, fab_fn_vals = [], []
        for idx, fn in enumerate(self.ehrlich_functions):
            if self.input_type == 'monospecific' and idx == 1:
                seqs = fab_sequences[0].clone()
            else:
                seqs = torch.tensor(sequence_sampler(fn.initial_sequence, self.edit_prob, self.num_samples))
            fab_sequences.append(seqs)
            fab_fn_vals.append(fn(seqs) + torch.randn(self.num_samples) * self.noise_level)

        # Add FC placeholders
        for _ in range(2 + (1 if self.input_type == 'monospecific' else 0)):
            fab_sequences.append(torch.zeros(self.num_samples, 297))
            fab_fn_vals.append(torch.zeros(self.num_samples))

        stacked_seqs = torch.stack(fab_sequences, dim=2)
        stacked_fns = torch.stack(fab_fn_vals, dim=1)
        del fab_sequences, fab_fn_vals

        # Determine connectivity types
        available = CONNECTIVITY_TYPES[self.input_type]
        if self.connectivity_filter is not None:
            filtered = [available[i] for i in self.connectivity_filter]
            indices = self.connectivity_filter
        else:
            filtered = available
            indices = list(range(len(available)))

        # Build data list
        data_list = []
        for idx in range(self.num_samples):
            fn_values = stacked_fns[idx]
            x = stacked_seqs[idx].T  # [seq_len, num_fabs] -> [num_fabs, seq_len]

            conn_idx = random.randint(0, len(filtered) - 1)
            edge_index = filtered[conn_idx]
            y = torch.tensor([complex_format_fn(fn_values, edge_index, complex_format=self.input_type)])

            data = Data(x=x, y=y, edge_index=edge_index, fn_values=fn_values)
            data.connectivity_type = indices[conn_idx]
            data_list.append(data)

        del stacked_seqs, stacked_fns
        gc.collect()
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
