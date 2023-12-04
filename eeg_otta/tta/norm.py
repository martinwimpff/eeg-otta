import mne
import torch
from torch import nn

from .alignment import OnlineAlignment
from .base import TTAMethod
from .bn import AlphaBatchNorm, RobustBN


class Norm(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(Norm, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))
        outputs = self.model(x)
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        if self.config.get("norm") == "norm_test":  # BN-1
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        elif self.config.get("norm") == "norm_alpha":  # BN-0.1
            self.model = AlphaBatchNorm.adapt_model(
                self.model, alpha=self.config.get("alpha"))
        elif self.config.get("norm") == "robust_norm":  # RoTTA
            self.model = RobustBN.adapt_model(
                self.model, alpha=self.config.get("alpha"))
        else:
            raise NotImplementedError
