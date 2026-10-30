"""Factorized Complementary Residual Mixture-of-Experts."""
import torch
from torch import nn
from torch.nn import functional as F


class FCRMoE(nn.Module):
    """Independent binary expert routing for low/high prior residuals.

    Each band mixes a no-op expert with a correction expert.  Keeping the
    historical ``body`` name makes validated spatial-gate checkpoints load
    directly; the forward pass makes the expert interpretation explicit.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 2, 3, padding=1))

    def route(self, low, baseline, prior):
        features = torch.cat((low, baseline, prior, (prior-baseline).abs()), 1)
        return torch.sigmoid(self.body(features))

    @staticmethod
    def residual_bands(baseline, prior):
        residual = prior - baseline
        low_residual = F.avg_pool2d(
            F.pad(residual, (15,) * 4, mode="reflect"), 31, 1)
        return low_residual, residual - low_residual

    @staticmethod
    def mix_experts(baseline, low_residual, high_residual, routes):
        # E_noop=0 and E_correct=R_b for each band b.  Written explicitly as
        # mixtures so the implementation matches the method definition.
        low_mixture = ((1-routes[:, :1]) * torch.zeros_like(low_residual) +
                       routes[:, :1] * low_residual)
        high_mixture = ((1-routes[:, 1:]) * torch.zeros_like(high_residual) +
                        routes[:, 1:] * high_residual)
        return (baseline + low_mixture + high_mixture).clamp(0, 1)

    def forward(self, low, baseline, prior, routes=None):
        if routes is None:
            routes = self.route(low, baseline, prior)
        low_residual, high_residual = self.residual_bands(baseline, prior)
        return self.mix_experts(
            baseline, low_residual, high_residual, routes), routes
