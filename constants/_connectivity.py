import torch
import itertools
from typing import Iterable, List, Tuple, Dict
import random

# set seed
random.seed(42)
torch.manual_seed(42)

def generate_formats(number_fab: int, number_fc: int = 2, fc2_threshold: int = 2) -> List[List[List[int]]]:
    fc1 = number_fab
    fc2 = number_fab + 1

    all_formats: List[List[List[int]]] = []

    def dfs(order: Tuple[int, ...],
            idx: int,
            tips: List[int],          # [tip_a, tip_b, tip_fc2]
            edges: List[List[int]],
            fc2_used: int):
        if idx == len(order):
            all_formats.append(edges)
            return

        node = order[idx]
        for slot, target in enumerate(tips):
            new_edges = edges + [[node, target]]
            new_tips = tips.copy()
            new_fc2_used = fc2_used

            if target == fc2:
                new_fc2_used += 1
                # fc2 turns into a chain *after* reaching the threshold
                if new_fc2_used >= fc2_threshold:
                    new_tips[slot] = node      # replace fc2 with the new node
                # else: keep pointing at fc2 (still hub-like)
            else:
                # Non-fc2 branch always chains
                new_tips[slot] = node

            dfs(order, idx + 1, new_tips, new_edges, new_fc2_used)

    # Enumerate every seed pair (a,b) attached to fc1
    for a, b in itertools.combinations(range(number_fab), 2):
        base_edges = [[fc1, fc2], [a, fc1], [b, fc1]]
        remaining = set(range(number_fab)) - {a, b}

        if not remaining:
            # Bispecific (no remaining fabs)
            all_formats.append(base_edges)
            continue

        # Try every permutation of the remaining fabs
        for order in itertools.permutations(remaining):
            tips = [a, b, fc2]  # a-branch tip, b-branch tip, fc2-slot
            dfs(order, 0, tips, base_edges, fc2_used=0)

    return all_formats


def canonicalize_graph(
    edges: Iterable[Iterable[int]],
    *,
    directed: bool = False
) -> Tuple[Tuple[int, int], ...]:
    """
    Turn an edge list into a canonical, hashable signature.
    - Removes duplicate edges.
    - If undirected: normalize (u, v) so u <= v.
    - Sorts edges to make order irrelevant.
    """
    if directed:
        norm = { (int(u), int(v)) for u, v in edges }  # keep direction
    else:
        norm = { (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges }  # undirected

    # Sort for stability and return as an immutable tuple
    return tuple(sorted(norm))

def dedupe_graphs_to_tensors(
    all_formats: List[List[List[int]]],
    *,
    directed: bool = False,
    dtype: torch.dtype = torch.long
) -> List[torch.Tensor]:
    """
    Deduplicate graphs by canonical signature and return a list of 2×E tensors.
    """
    seen = set()
    unique_tensors: List[torch.Tensor] = []

    for edges in all_formats:
        sig = canonicalize_graph(edges, directed=directed)
        if sig in seen:
            continue
        seen.add(sig)

        # Convert the canonical signature (sorted tuples) to a 2×E tensor
        if len(sig) == 0:
            t = torch.empty((2, 0), dtype=dtype)
        else:
            arr = torch.tensor(sig, dtype=dtype)   # shape: (E, 2)
            t = arr.t().contiguous()               # shape: (2, E)
        unique_tensors.append(t)

    return unique_tensors

CONNECTIVITY_TYPES = {
        'pentaspecific': dedupe_graphs_to_tensors(generate_formats(number_fab=5, number_fc=2, fc2_threshold=2), directed=False, dtype=torch.long),
        'tetraspecific': dedupe_graphs_to_tensors(generate_formats(number_fab=4, number_fc=2, fc2_threshold=2), directed=False, dtype=torch.long),
        'trispecific': dedupe_graphs_to_tensors(generate_formats(number_fab=3, number_fc=2, fc2_threshold=2), directed=False, dtype=torch.long),
        'bispecific': dedupe_graphs_to_tensors(generate_formats(number_fab=2, number_fc=2, fc2_threshold=2), directed=False, dtype=torch.long),
        'monospecific': dedupe_graphs_to_tensors(generate_formats(number_fab=2, number_fc=2, fc2_threshold=2), directed=False, dtype=torch.long),
    }

trispecific_example = [
    torch.tensor([[0, 1, 1, 3, 3, 2, 4, 3], [1, 0, 3, 1, 2, 3, 3, 4]], dtype=torch.long), 
    torch.tensor([[2, 1, 1, 3, 3, 0, 4, 3], [1, 2, 3, 1, 0, 3, 3, 4]], dtype=torch.long)
]

CONNECTIVITY_TYPES['trispecific_example'] = trispecific_example


CONNECTIVITY_WEIGHTS = {

    'pentaspecific': [torch.tensor([random.uniform(1, 10) for i in range(7)], dtype=torch.float) for i in range(len(CONNECTIVITY_TYPES['pentaspecific']))], # fabs  + 2*fc
    'tetraspecific': [torch.tensor([random.uniform(1, 10) for i in range(6)], dtype=torch.float) for i in range(len(CONNECTIVITY_TYPES['tetraspecific']))],
    'trispecific': [torch.tensor([random.uniform(1, 10) for i in range(5)], dtype=torch.float) for i in range(len(CONNECTIVITY_TYPES['trispecific']))],
    'bispecific': [torch.tensor([random.uniform(1, 10) for i in range(4)], dtype=torch.float) for i in range(len(CONNECTIVITY_TYPES['bispecific']))],
    'trispecific_example': [torch.tensor([random.uniform(1, 10) for i in range(5)], dtype=torch.float) for i in range(len(trispecific_example))],
}

CONNECTIVITY_WEIGHTS['monospecific'] = CONNECTIVITY_WEIGHTS['bispecific'] # make them the same



def get_connectivity_weight(complex_format: str, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Get the weight tensor for a given edge_index by finding its position in CONNECTIVITY_TYPES.
    
    Args:
        complex_format: 'trispecific' or 'bispecific'
        edge_index: The edge index tensor to find weights for
    
    Returns:
        The corresponding weight tensor
    
    Raises:
        ValueError: If the edge_index is not found in the connectivity types
    """
    connectivity_list = CONNECTIVITY_TYPES[complex_format]
    
    # Find the index of the matching edge_index
    for i, connectivity in enumerate(connectivity_list):
        if torch.equal(edge_index, connectivity):
            return CONNECTIVITY_WEIGHTS[complex_format][i]
    
    raise ValueError(f"Edge index not found in {complex_format} connectivity types")

def sample_connectivity_with_weights(complex_format: str):
    """
    Sample a random connectivity type and return both edge_index and weights.
    
    Args:
        complex_format: 'trispecific' or 'bispecific'
    
    Returns:
        Tuple of (edge_index, weights)
    """
    import random
    
    connectivity_list = CONNECTIVITY_TYPES[complex_format]
    weights_list = CONNECTIVITY_WEIGHTS[complex_format]
    
    # Sample random index
    idx = random.randint(0, len(connectivity_list) - 1)
    
    return connectivity_list[idx], weights_list[idx]

