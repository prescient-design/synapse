import sys
from copy import deepcopy
from pathlib import Path

# Convert wandb args to Hydra args
sys.argv = [a[2:] if a.startswith('--') and '=' in a else a for a in sys.argv]

_DIR = Path(__file__).parent.resolve()
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import functools

import hydra
import lightning
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

# Allow common classes in checkpoint loading (PyTorch 2.6+ defaults to weights_only=True)
torch.serialization.add_safe_globals([functools.partial, Adam, ReduceLROnPlateau, CosineAnnealingLR, StepLR])
import einops
from lightning import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

from synthetic_dataset import AntibodySyntheticDataset


class PretrainingModule(LightningModule):
    """Pretrain GNN on node-level function values."""

    def __init__(self, gnn, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['gnn'])
        self.gnn = gnn
        self.head = nn.Sequential(nn.Linear(gnn.emb_dim, gnn.emb_dim), nn.LayerNorm(gnn.emb_dim),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(gnn.emb_dim, 1))
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        x = self.gnn.token_embedding(batch.x.long())
        for block in self.gnn.transformer_blocks:
            x = block(x)
        x = einops.reduce(x, "n s e -> n e", "mean")
        if self.gnn.use_connectivity_embedding and hasattr(batch, 'connectivity_type'):
            x = torch.cat([x, self.gnn.connectivity_embedding(batch.connectivity_type.long())[batch.batch]], dim=-1)
        for layer in self.gnn.layers:
            x = layer(x, batch.edge_index)
        return self.head(x)

    def training_step(self, batch, _):
        t = batch.fn_values.unsqueeze(-1) if batch.fn_values.dim() == 1 else batch.fn_values
        loss = self.loss_fn(self(batch), t)
        self.log("pretrain/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        t = batch.fn_values.unsqueeze(-1) if batch.fn_values.dim() == 1 else batch.fn_values
        self.log("pretrain/val_loss", self.loss_fn(self(batch), t), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def transfer_weights(src, tgt, freeze=False):
    """Copy weights from pretrained model."""
    s, t = src.gnn, getattr(tgt, "module", tgt)
    with torch.no_grad():
        if hasattr(s, 'token_embedding') and hasattr(t, 'token_embedding'):
            w = s.token_embedding.TokenEmbeddings.data
            t.token_embedding.TokenEmbeddings.data.copy_(w)
        if hasattr(s, 'transformer_blocks'):
            t.transformer_blocks.load_state_dict(s.transformer_blocks.state_dict())
        if hasattr(s, 'layers') and len(s.layers) == len(t.layers):
            for sl, tl in zip(s.layers, t.layers):
                tl.load_state_dict(sl.state_dict())
        if s.connectivity_embedding and t.connectivity_embedding:
            t.connectivity_embedding.load_state_dict(s.connectivity_embedding.state_dict())
    if freeze:
        for n in ['token_embedding', 'transformer_blocks', 'layers', 'connectivity_embedding']:
            if hasattr(t, n) and getattr(t, n):
                for p in getattr(t, n).parameters():
                    p.requires_grad = False


def make_loaders(cfg, pool_size, n_pre, n_fine, batch_size, seed):
    dm = hydra.utils.instantiate(cfg.datamodule)
    c = dm.config
    ds = AntibodySyntheticDataset(num_states=c.num_states, num_motifs=c.num_motifs, motif_length=c.motif_length, initial_seed=c.seed,
                                   num_samples=pool_size, input_type=c.input_type,
                                   edit_prob=c.edit_prob, noise_level=c.noise_level)
    
    def split(d):
        v = int(0.1 * len(d))
        return random_split(d, [len(d) - v, v], generator=torch.Generator().manual_seed(seed))
    
    pre_tr, pre_val = split(Subset(ds, list(range(n_pre))))
    ft_tr, ft_val = split(Subset(ds, list(range(min(n_fine, n_pre)))))
    kw = dict(batch_size=batch_size, num_workers=c.num_workers, pin_memory=c.pin_memory)
    return DataLoader(pre_tr, shuffle=True, **kw), DataLoader(pre_val, **kw), \
           DataLoader(ft_tr, shuffle=True, **kw), DataLoader(ft_val, **kw)


def make_trainer(cfg, monitor="Loss (Validation)"):
    cbs = [hydra.utils.instantiate(c) for c in (cfg.get("callbacks") or {}).values()
           if isinstance(c, DictConfig) and "_target_" in c]
    for cb in cbs:
        if isinstance(cb, (EarlyStopping, ModelCheckpoint)):
            cb.monitor, cb.mode = monitor, "min"
    logs = [hydra.utils.instantiate(l) for l in (cfg.get("logger") or {}).values()
            if isinstance(l, DictConfig) and "_target_" in l]
    return hydra.utils.instantiate(cfg.trainer, callbacks=cbs, logger=logs)


@hydra.main(config_name="train_with_pretraining.yaml", config_path="configurations", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.datamodule, 'input_type') and hasattr(cfg.module.module, 'connectivity_format'):
        cfg.module.module.connectivity_format = cfg.datamodule.input_type
    OmegaConf.set_struct(cfg, True)

    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    p = cfg.pretraining
    pre_tr, pre_val, ft_tr, ft_val = make_loaders(
        cfg, p.get("pool_size", 50000), p.num_samples, p.get("finetune_num_samples", p.num_samples),
        p.batch_size, cfg.get("seed", 42))

    if p.enabled:
        # Pretrain
        gnn = hydra.utils.instantiate(cfg.module.module)
        pre_mod = PretrainingModule(gnn, lr=p.lr)
        pre_cfg = deepcopy(cfg)
        if p.get("max_epochs"):
            pre_cfg.trainer.max_epochs = p.max_epochs
        make_trainer(pre_cfg, "pretrain/val_loss").fit(pre_mod, train_dataloaders=pre_tr, val_dataloaders=pre_val)

        # Fine-tune
        model = hydra.utils.instantiate(cfg.module)
        transfer_weights(pre_mod, model, freeze=p.get("freeze_pretrained_weights", False))
        trainer = make_trainer(cfg)
        trainer.fit(model, train_dataloaders=ft_tr, val_dataloaders=ft_val)
        if cfg.get("test"):
            trainer.test(model, dataloaders=ft_val, ckpt_path=trainer.checkpoint_callback.best_model_path or None)
    else:
        model = hydra.utils.instantiate(cfg.module)
        trainer = make_trainer(cfg)
        trainer.fit(model, train_dataloaders=ft_tr, val_dataloaders=ft_val)
        if cfg.get("test"):
            trainer.test(model, dataloaders=ft_val, ckpt_path=trainer.checkpoint_callback.best_model_path or None)


if __name__ == "__main__":
    main()
