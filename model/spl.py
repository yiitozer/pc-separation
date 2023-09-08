import torch
from torch import nn
from typing import List

from model.unet import UNet


class Spleeter(nn.Module):
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 in_channels: int = 2,
                 stft_window_size: int = 4096,
                 encoder_act: str = 'relu',
                 decoder_act: str = 'elu',
                 conv_n_filters: List = None,
                 decoder_dropout: bool = True,
                 kernel_size: int = 5,
                 stride: int = 2,
                 separation_exponent: int = 2,
                 targets: List = None,
                 max_freq_bins: int = 1024,
                 max_time_frames: int = 512):

        super(Spleeter, self).__init__()

        if targets is None:
            targets = ['piano', 'orch']

        self._device = device
        self._separation_exponent = separation_exponent
        self._stft_window_size = stft_window_size
        self._max_freq_bins = max_freq_bins
        self._max_time_frames = max_time_frames
        self._targets = targets

        self._unets = nn.ModuleList([UNet(device=device,
                                          in_channels=in_channels,
                                          window_size=stft_window_size,
                                          encoder_act=encoder_act,
                                          decoder_act=decoder_act,
                                          conv_n_filters=conv_n_filters,
                                          decoder_dropout=decoder_dropout,
                                          kernel_size=kernel_size,
                                          stride=stride).to(self._device) for _ in range(len(self._targets))])

        self.apply(_weights_init)

        if conv_n_filters is None:
            self._conv_n_filters = [16, 32, 64, 128, 256, 512]
        else:
            self._conv_n_filters = conv_n_filters

        self._masks = dict()

    def forward(self, X_c):
        # X_c.shape = [B, C, T, F]
        output_dict = {target: self._unets[idx](X_c) for idx, target in enumerate(self._targets)}

        return output_dict


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)