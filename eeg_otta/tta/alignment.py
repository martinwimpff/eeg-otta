import mne
import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy import linalg
import torch
from torch import nn

from .base import TTAMethod


class OnlineAlignment(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(OnlineAlignment, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        x_aligned = self.align_data(
            x, self.config.get("alignment"),
            self.config.get("averaging_method", "equal"),
            self.config.get("alpha", None))
        outputs = self.model(x_aligned)
        return outputs

    @staticmethod
    def align_data(x, alignment, averaging_method: str, alpha: float = None):
        n_trials = x.shape[0]
        weights = OnlineAlignment._calculate_weights(n_trials, averaging_method, alpha)
        covmats = torch.matmul(x, x.transpose(1, 2)).detach().cpu().numpy()
        if alignment == "euclidean":
            R = np.average(covmats, axis=0, weights=weights)
        elif alignment == "riemann":
            R = mean_riemann(covmats, sample_weight=weights)
        else:
            raise NotImplementedError
        R_op = linalg.inv(linalg.sqrtm(R))
        x_aligned = torch.matmul(
            torch.tensor(R_op, dtype=torch.float32, device=x.device), x)
        return x_aligned

    @staticmethod
    def _calculate_weights(n_trials: int, averaging_method: str, alpha: float = None):
        if averaging_method == "equal":
            weights = None
        elif averaging_method == "zanini":
            weights = np.arange(1, n_trials + 1) / n_trials
        elif averaging_method == "ema":
            assert alpha is not None
            if n_trials == 1:
                weights = np.array([1.])
            else:
                first, last = (1 - alpha) ** (n_trials - 1), alpha
                if n_trials >= 3:
                    weights = [alpha * ((1 - alpha) ** i) for i in reversed(
                        range(1, n_trials - 1))]
                    weights = [first] + weights + [last]
                else:
                    weights = [first, last]
                weights = np.array(weights)
        else:
            raise NotImplementedError

        return weights

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
