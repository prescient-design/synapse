"""Lightning module for training GNN models."""

import torch
from lightning import LightningModule
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef


class SynapseLightningModule(LightningModule):
    def __init__(self, module, optimizer, scheduler):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["module"])
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = MSELoss()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.test_mse = MeanSquaredError()
        self.test_pearson = PearsonCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

    def forward(self, x):
        return self.module(x)

    def _step(self, batch):
        x, edge_index, batch_idx, y = batch.x, batch.edge_index, batch.batch, batch.y
        conn = getattr(batch, 'connectivity_type', None)
        out = self.module(x, edge_index, batch_idx, connectivity_type=conn)
        y = y.unsqueeze(-1) if y.dim() == 1 else y
        return self.loss_fn(out, y), out, y

    def training_step(self, batch, _):
        loss, _, _ = self._step(batch)
        self.log("Loss (Train)", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, pred, target = self._step(batch)
        self.log("Loss (Validation)", loss, on_epoch=True, prog_bar=True)
        self.val_mse(pred, target)
        self.val_pearson(pred.squeeze(-1), target.squeeze(-1))
        self.val_spearman(pred.squeeze(-1), target.squeeze(-1))
        self.log("Validation MSE", self.val_mse, on_epoch=True)
        self.log("Validation Pearson", self.val_pearson, on_epoch=True)
        self.log("Validation Spearman", self.val_spearman, on_epoch=True)

    def test_step(self, batch, _):
        _, pred, target = self._step(batch)
        self.test_mse(pred, target)
        self.test_pearson(pred.squeeze(-1), target.squeeze(-1))
        self.test_spearman(pred.squeeze(-1), target.squeeze(-1))
        self.log("Test MSE", self.test_mse, on_epoch=True, prog_bar=True)
        self.log("Test Pearson", self.test_pearson, on_epoch=True)
        self.log("Test Spearman", self.test_spearman, on_epoch=True)

    def configure_optimizers(self):
        opt = self.hparams.optimizer(params=self.module.parameters())
        return {"optimizer": opt, "lr_scheduler": {
            "scheduler": self.scheduler(optimizer=opt), "monitor": "Loss (Validation)", "interval": "epoch"
        }}
