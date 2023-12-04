from copy import deepcopy

import mne
import torch
from torch import nn


class TTAMethod(nn.Module):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(TTAMethod, self).__init__()
        self.model = model
        self.config = config
        self.info = info
        self.device = self.model.device
        self.input_buffer = None
        self.buffer_length = self.config.get("buffer_length")
        self.buffer_counter = 0

        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.print_amount_trainable_params()

    def forward(self, x):

        if x.shape[0] == 1:  # Only single-sample test-time adaptation allowed

            # add sample to buffer, replace the oldest sample if buffer is full
            if self.input_buffer is None:
                self.input_buffer = x
            elif self.input_buffer.shape[0] < self.buffer_length:
                self.input_buffer = torch.cat([self.input_buffer, x], dim=0)
            else:
                self.input_buffer = torch.cat([self.input_buffer[1:], x], dim=0)

            # update the model if the complete buffer has changed
            if self.buffer_counter == (self.buffer_length - 1):
                outputs = self.forward_and_adapt(self.input_buffer)
                outputs = outputs[-1].unsqueeze(0)
            else:
                outputs = self.forward_sliding_window(self.input_buffer)
                outputs = outputs[-1].unsqueeze(0)

            # increase counter
            self.buffer_counter += 1
            self.buffer_counter %= self.buffer_length

        else:
            outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        return self.model(x)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.config["optimizer"] == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.config["optimizer_kwargs"]["lr"],
                                    betas=(self.config["optimizer_kwargs"]["beta"], 0.999),
                                    weight_decay=self.config["optimizer_kwargs"]["weight_decay"])
        else:
            raise NotImplementedError

    def print_amount_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        print(f"#Trainable/total parameters: {trainable}/{total}")

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state
