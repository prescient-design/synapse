"""Metrics for model evaluation."""

from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

regression_metrics = {
    "MSE": MeanSquaredError(),
    "Pearson": PearsonCorrCoef(),
    "Spearman": SpearmanCorrCoef(),
}
