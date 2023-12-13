import torch
import torch.nn as nn


class Separator(nn.Module):
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 model_cfg = None):
        super(Separator, self).__init__()
        self._device = device
        self._model_cfg = model_cfg
        self._sample_rate = model_cfg.audio.sample_rate
        self._num_channels = 1 if model_cfg.audio.mono else 2

    def load_model(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()
