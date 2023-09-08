import torch
from torch import nn
from typing import List

from model.modules import Conv2dSame


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 kernel_size: int,
                 stride: int,
                 act: nn.Module = nn.ReLU()):
        super(EncoderBlock, self).__init__()

        self._conv = Conv2dSame(in_filters,
                                out_filters,
                                kernel_size=kernel_size,
                                stride=stride)

        self._bn = nn.BatchNorm2d(out_filters,
                                  track_running_stats=True,
                                  eps=1e-3,
                                  momentum=0.01)

        self._act = act

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._act(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 kernel_size: int,
                 stride: int,
                 act: nn.Module = nn.ELU(alpha=1),
                 dropout: bool = True):
        super(DecoderBlock, self).__init__()

        self._deconv = nn.ConvTranspose2d(in_filters,
                                          out_filters,
                                          padding=2,
                                          output_padding=1,
                                          kernel_size=kernel_size,
                                          stride=stride)

        self._bn = nn.BatchNorm2d(out_filters,
                                  track_running_stats=True,
                                  eps=1e-3,
                                  momentum=0.01)
        self._act = act

        if dropout:
            self._dropout = nn.Dropout(0.5)
        else:
            self._dropout = None

    def forward(self, x):
        x = self._deconv(x)
        x = self._act(x)
        x = self._bn(x)

        if self._dropout is not None:
            x = self._dropout(x)

        return x


class UNet(nn.Module):
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 in_channels: int = 2,
                 window_size: int = 4096,
                 encoder_act: str = 'relu',
                 decoder_act: str = 'elu',
                 conv_n_filters: List = None,
                 decoder_dropout: bool = True,
                 kernel_size: int = 5,
                 stride: int = 2):
        super(UNet, self).__init__()
        self._device = device
        self._in_channels = in_channels
        self._encoder_act = self._get_activation_layer(encoder_act)
        self._decoder_act = self._get_activation_layer(decoder_act)
        self._decoder_dropout = decoder_dropout
        self._kernel_size = kernel_size
        self._stride = stride
        self.win = nn.Parameter(torch.hann_window(window_size), requires_grad=False)

        if conv_n_filters is None:
            self._conv_n_filters = [16, 32, 64, 128, 256, 512]
        else:
            self._conv_n_filters = conv_n_filters

        self._enc1 = EncoderBlock(self._in_channels,
                                  self._conv_n_filters[0],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._enc2 = EncoderBlock(self._conv_n_filters[0],
                                  self._conv_n_filters[1],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._enc3 = EncoderBlock(self._conv_n_filters[1],
                                  self._conv_n_filters[2],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._enc4 = EncoderBlock(self._conv_n_filters[2],
                                  self._conv_n_filters[3],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._enc5 = EncoderBlock(self._conv_n_filters[3],
                                  self._conv_n_filters[4],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._enc6 = EncoderBlock(self._conv_n_filters[4],
                                  self._conv_n_filters[5],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._encoder_act)

        self._dec1 = DecoderBlock(self._conv_n_filters[5],
                                  self._conv_n_filters[4],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._dec2 = DecoderBlock(self._conv_n_filters[5],
                                  self._conv_n_filters[3],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._dec3 = DecoderBlock(self._conv_n_filters[4],
                                  self._conv_n_filters[2],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._dec4 = DecoderBlock(self._conv_n_filters[3],
                                  self._conv_n_filters[1],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._dec5 = DecoderBlock(self._conv_n_filters[2],
                                  self._conv_n_filters[0],
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._dec6 = DecoderBlock(self._conv_n_filters[1],
                                  1,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  act=self._decoder_act)

        self._last_layer = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=self._in_channels,
                                                   kernel_size=4,
                                                   dilation=2,
                                                   padding=3),
                                         nn.Sigmoid())

    def forward(self, x):
        enc1 = self._enc1(x)
        enc2 = self._enc2(enc1)
        enc3 = self._enc3(enc2)
        enc4 = self._enc4(enc3)
        enc5 = self._enc5(enc4)
        enc6 = self._enc6(enc5)

        dec1 = self._dec1(enc6)
        merge1 = torch.cat((enc5, dec1), dim=1)
        dec2 = self._dec2(merge1)
        merge2 = torch.cat((enc4, dec2), dim=1)
        dec3 = self._dec3(merge2)
        merge3 = torch.cat((enc3, dec3), dim=1)
        dec4 = self._dec4(merge3)
        merge4 = torch.cat((enc2, dec4), dim=1)
        dec5 = self._dec5(merge4)
        merge5 = torch.cat((enc1, dec5), dim=1)
        dec6 = self._dec6(merge5)

        # Last layer to ensure initial shape reconstruction.
        out = self._last_layer(dec6)

        return out

    @staticmethod
    def _get_activation_layer(act_type: str = 'relu') -> nn.Module:
        if act_type.lower() == 'relu':
            return nn.ReLU()

        elif act_type.lower() == 'elu':
            return nn.ELU(alpha=1)

        elif act_type.lower() == 'leakyrelu':
            return nn.LeakyReLU(0.2)

        else:
            raise ValueError(f'{act_type} is invalid! Please choose among ReLU, ELU or LeakyReLU.')


if __name__ == '__main__':
    net = UNet(2)
    print(net(torch.rand(1, 2, 1024, 512)).shape)
