AMINO_ACIDS = '-ACDEFGHIKLMNPQRSTVWY'
amino_acids = list(AMINO_ACIDS)
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}

__all__ = [
    "aa_to_idx",
    "idx_to_aa",
]