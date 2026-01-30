"""Lightning DataModule for synthetic dataset."""

from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.utils.data
from lightning import LightningDataModule
from torch import Generator
from torch_geometric.loader import DataLoader

from synthetic_dataset import AntibodySyntheticDataset


@dataclass
class DataModuleConfig:
    root: Union[str, Path] = "data/"
    num_fabs: int = 3
    num_states: int = 20
    num_motifs: int = 2
    motif_length: int = 3
    num_samples: int = 100
    seed: int = 0x00B1
    lengths: Tuple[Number, Number, Number] = (0.8, 0.1, 0.1)
    fixed_test_size: Optional[int] = None
    test_different_motifs: bool = False
    test_motif_seed_offset: int = 10000
    noise_level: float = 0.01
    edit_prob: float = 0.1
    input_type: str = 'trispecific'
    connectivity_split: bool = False
    train_connectivity_types: Optional[Tuple[int, ...]] = None
    test_connectivity_types: Optional[Tuple[int, ...]] = None
    epistasis_factor: float = 0.0
    batch_size: int = 32
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False

    def __post_init__(self):
        if self.fixed_test_size is None:
            if abs(sum(self.lengths) - 1.0) > 1e-6:
                raise ValueError(f"Split lengths must sum to 1.0, got {sum(self.lengths)}")
        elif self.fixed_test_size <= 0:
            raise ValueError(f"fixed_test_size must be positive")

        valid_types = ['monospecific', 'trispecific', 'trispecific_example', 'bispecific', 'pentaspecific', 'tetraspecific']
        if self.input_type not in valid_types:
            raise ValueError(f"Invalid input_type: {self.input_type}")

        fabs_map = {'trispecific': 3, 'bispecific': 2, 'pentaspecific': 5, 'tetraspecific': 4, 'monospecific': 1, 'trispecific_example': 3}
        self.num_fabs = fabs_map.get(self.input_type, 3)

        if self.connectivity_split:
            if self.train_connectivity_types is None or self.test_connectivity_types is None:
                raise ValueError("When connectivity_split=True, both train/test connectivity_types must be specified")
            if self.fixed_test_size is None:
                self.lengths = (1.0, 0.0, 0.0)


class SynapseDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = DataModuleConfig(**kwargs)
        self.save_hyperparameters(logger=False)
        generator = kwargs.get('generator') or Generator()
        self.generator = generator.manual_seed(self.config.seed)
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def _generate_ehrlich_functions(self, seed_offset=0):
        from holo.test_functions.closed_form import Ehrlich
        base_seed = self.config.seed + seed_offset
        return [Ehrlich(num_states=self.config.num_states, num_motifs=self.config.num_motifs,
                       motif_length=self.config.motif_length, dim=297,
                       random_seed=self.config.seed if self.config.input_type == 'monospecific' else i + base_seed,
                       epistasis_factor=self.config.epistasis_factor)
                for i in range(self.config.num_fabs)]

    def _create_dataset(self, ehrlich_fns=None, num_samples=None, connectivity_filter=None):
        return AntibodySyntheticDataset(
            num_states=self.config.num_states, num_motifs=self.config.num_motifs,
            motif_length=self.config.motif_length, initial_seed=self.config.seed,
            num_samples=num_samples or self.config.num_samples, input_type=self.config.input_type,
            edit_prob=self.config.edit_prob, noise_level=self.config.noise_level,
            ehrlich_functions=ehrlich_fns, connectivity_filter=connectivity_filter)

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        print(f"Creating {self.config.input_type} dataset with {self.config.num_samples} samples")

        if self.config.fixed_test_size is not None:
            shared = self._generate_ehrlich_functions()
            test_ehrlich = self._generate_ehrlich_functions(self.config.test_motif_seed_offset) if self.config.test_different_motifs else shared
            self.test_dataset = self._create_dataset(test_ehrlich, self.config.fixed_test_size)
            train_val = self._create_dataset(shared)
            val_size = max(1, int(0.1 * len(train_val)))
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_val, [len(train_val) - val_size, val_size], generator=self.generator)

        elif self.config.connectivity_split:
            self.train_dataset = self._create_dataset(connectivity_filter=list(self.config.train_connectivity_types))
            self.test_dataset = self._create_dataset(connectivity_filter=list(self.config.test_connectivity_types))
            val_size = int(0.1 * len(self.train_dataset))
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [len(self.train_dataset) - val_size, val_size], generator=self.generator)

        else:
            full = self._create_dataset()
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                full, self.config.lengths, generator=self.generator)

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")

        try:
            from holo.test_functions.closed_form._ehrlich import clear_oas_cache
            clear_oas_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                          num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
                          persistent_workers=self.config.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
                          persistent_workers=self.config.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
                          persistent_workers=self.config.persistent_workers)
