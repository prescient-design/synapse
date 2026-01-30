#!/usr/bin/env python3
"""Training script for wandb sweeps."""

import sys
from pathlib import Path

# Convert wandb args (--param=value) to Hydra args (param=value)
sys.argv = [a[2:] if a.startswith('--') and '=' in a else a for a in sys.argv]

_DIR = Path(__file__).parent.resolve()
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import hydra
import lightning
import torch
from omegaconf import DictConfig, OmegaConf

# PyTorch 2.6+ compatibility
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})


@hydra.main(config_name="train.yaml", config_path="configurations", version_base="1.3")
def main(cfg: DictConfig):
    # Sync connectivity_format with input_type
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg.datamodule, 'input_type') and hasattr(cfg.module.module, 'connectivity_format'):
        cfg.module.module.connectivity_format = cfg.datamodule.input_type
    OmegaConf.set_struct(cfg, True)

    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.module)
    
    callbacks = [hydra.utils.instantiate(c) for c in (cfg.get("callbacks") or {}).values() 
                 if isinstance(c, DictConfig) and "_target_" in c]
    loggers = [hydra.utils.instantiate(l) for l in (cfg.get("logger") or {}).values()
               if isinstance(l, DictConfig) and "_target_" in l]

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if trainer.logger:
        for lg in trainer.loggers:
            lg.log_hyperparams({"datamodule": cfg.datamodule, "module": cfg.module, "seed": cfg.get("seed")})

    if cfg.get("train"):
        trainer.fit(model, datamodule=datamodule)

    if cfg.get("test"):
        trainer.test(model, ckpt_path=trainer.checkpoint_callback.best_model_path or None, datamodule=datamodule)

    try:
        import wandb
        if wandb.run:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
