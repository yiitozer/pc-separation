import json
import math
from pathlib import Path
import torch
from torch.nn import Module, L1Loss, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Dict

from dsp.filtering import wiener
from solver import Solver
from model.spl import Spleeter
from utils import AverageMeter, Config

EPSILON = 1e-10


class SPLCaCSolver(Solver):
    def __init__(self,
                 model_type: str = 'spl_cac',
                 device: torch.device = torch.device('cuda'),
                 optimizer_func: torch.optim = torch.optim.AdamW,
                 model_cfg: Config = None,
                 dataset_dir: str = '',
                 output_dir: str = '',
                 train_folder_name: str = 'train_aug',
                 val_folder_name: str = 'val_aug',
                 train: bool = True,
                 dataloader_kwargs: dict = {},
                 loss_func: Module = L1Loss()):

        super(SPLCaCSolver, self).__init__(model_type, device, optimizer_func, model_cfg, dataset_dir, output_dir,
                                           train_folder_name, val_folder_name, train, dataloader_kwargs)

        self._targets = model_cfg.targets
        self._max_time_frames = model_cfg.model.max_time_frames
        self._max_freq_bins = model_cfg.model.max_freq_bins
        self._window_size = model_cfg.stft.window_size
        self._separation_exponent = model_cfg.model.separation_exponent
        self._window = Parameter(torch.hann_window(self._window_size)).to(self._device)

        self._model = Spleeter(device=self._device,
                               in_channels=self._num_channels * 2,
                               stft_window_size=model_cfg.stft.window_size,
                               encoder_act=model_cfg.model.encoder_act,
                               decoder_act=model_cfg.model.decoder_act,
                               conv_n_filters=None,
                               decoder_dropout=model_cfg.model.decoder_dropout,
                               kernel_size=model_cfg.model.kernel_size,
                               stride=model_cfg.model.stride,
                               max_freq_bins=model_cfg.model.max_freq_bins,
                               max_time_frames=model_cfg.model.max_time_frames).to(self._device)

        if not train:
            self._model.eval()

        self._optimizer = self._optimizer_func(self._model.parameters(),
                                               lr=model_cfg.train.lr,
                                               eps=model_cfg.train.eps)
        self._loss_func = loss_func

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            factor=self._model_cfg.train.lr_decay_gamma,
            patience=self._model_cfg.train.lr_decay_patience,
            cooldown=10,
        )

    def _pad_and_partition(self,
                           spec: torch.Tensor):
        orig_num_time_frames = spec.size(-1)
        new_num_time_frames = math.ceil(orig_num_time_frames / self._max_time_frames) * self._max_time_frames
        spec = F.pad(spec, [0, new_num_time_frames - orig_num_time_frames])

        # Partition the tensor into multiple samples of length T and stack them into a batch
        return torch.cat(torch.split(spec, self._max_time_frames, dim=-1), dim=0)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input waveform

        Returns
        -------
        X_c_padded: torch.Tensor
            Complex spectrogram after padding and partitioning
        """

        # Number of samples in audio waveform
        batch_size, num_channels, num_time_steps = x.shape

        x = x.view(-1, num_time_steps)  # new shape: [batch_size * num_channels, num_time_steps]

        X_c = torch.stft(
            x,
            n_fft=self._model_cfg.stft.window_size,
            hop_length=self._model_cfg.stft.hop_size,
            window=self._window,
            center=self._model_cfg.stft.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )  # X_c.shape = [batch_size * num_channels, num_freq_bins, num_time_frames]

        _, num_freq_bins, num_time_frames = X_c.shape
        X_c = X_c.view(batch_size, num_channels, num_freq_bins, num_time_frames)  # B, C, F, T

        return X_c

    def _build_spec(self,
                    input_stft: torch.Tensor,
                    spec_exponent: float = 1.0,
                    window_exponent: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Builds spectrograms for training

        Parameters
        ----------
        input_stft: torch.Tensor [B, C, F, T]
            Complex input spectrogram, see self._stft

        spec_exponent: float
            Spectrogram exponent

        window_exponent: float
            Window exponent

        Returns
        -------
        X_c_padded: torch.Tensor [B, C, T, F]
            Padded, partitioned complex input spectrogram

        X_mag: torch.Tensor [B, C, T, F]
            Padded, partitioned magnitude spectrogram
        """

        # Crop to _max_fre_bins (1024 in the original implementation)
        # Transpose to B, C, T, F (needed input dims for training)
        X_c_cropped = input_stft[:, :, :self._max_freq_bins, :]

        # Pad and create partitions based on the allowed maximum time frames (512 in the original implementation)
        X_c_padded = self._pad_and_partition(X_c_cropped).transpose(2, 3) ** window_exponent

        # Compute magnitude spectrogram of the padded and partitioned complex input spectrogram
        X_mag = X_c_padded.abs() ** spec_exponent

        return X_c_padded, X_mag

    def _extend_mask(self,
                     mask):
        """Mask extension

        Note that the model is trained for an interval of frequencies [0, self._max_freq_bins (default: 1024)].
        At inference time, a zero padding is applied in the "cropped" region. This function assumes that the time bins
        are in the 3rd dimension (TODO: Is that a good idea?)

        Parameters
        ----------
        mask: torch.Tensor [B, C, F, self._max_time_frames]
            Estimated mask

        Returns
        -------
        torch.Tensor [B, C, F, self._window_size // 2 + 1]
            Mask with zero padding.

        """
        mask_shape = mask.shape  # B, C, F, T
        extension_row = torch.zeros((mask_shape[0], mask_shape[1], mask_shape[2], 1), device=mask.device)

        # Number of rows of the zero padded region
        n_extra_row = self._window_size // 2 + 1 - self._max_freq_bins
        extension = torch.tile(extension_row, [1, 1, 1, n_extra_row])

        return torch.cat((mask, extension), axis=3)

    def _build_mask_dict(self,
                         model_outputs: dict):
        """Builds a dictionary of masks. The items of the dictionary are the target instruments (e.g., piano and
            orchestra)

        Parameters
        ----------
        model_outputs: dict
            Dictionary of model outputs (estimated masks)

        Returns
        -------
        mask_dict: dict
            Dictionary of masks
        """

        mask_dict = dict()

        # The estimated masks are supposed to sum to 1, given a separation exponent (default:2)
        output_sum = sum([m ** self._separation_exponent for m in model_outputs.values()]) + EPSILON

        for target in self._targets:
            output = model_outputs[target]
            target_mask = (output ** self._separation_exponent + (EPSILON / len(model_outputs))) / output_sum
            mask_dict[target] = target_mask

        return mask_dict

    @staticmethod
    def _build_masked_stfts(input_stft: torch.Tensor,
                            mask_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Builds a dictionary of masked STFT spectrograms, given a complex input spectrogram.

        Parameters
        ----------
        input_stft: torch.Tensor [B, C, F, T]
            STFT matrix, first two channels are real and the latter two channels are imaginary.

        mask_dict: dict
            Dictionary of estimated masks (see self._build_mask_dict)

        Returns
        -------
        masked_stft_dict: dict
            Dictionary of padded and partitioned masked stfts-
        """

        masked_stft_dict = dict()
        for target in mask_dict:
            masked_stft_dict[target] = input_stft * mask_dict[target]

        return masked_stft_dict

    def _build_gt_spec_dict(self,
                            waveform_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Builds a dictionary of GT magnitude spectrograms

        Parameters
        ---------
        waveform_dict: dict
            Dictionary of target GT waveforms

        Returns
        -------
        spec_dict: dict
            Dictionary of target GT magnitude spectrograms
        """
        spec_dict = dict()

        for target in self._targets:
            X_c = self._build_spec(self._stft(waveform_dict[target].to(self._device)))[0]
            spec_dict[target] = torch.cat([X_c.real, X_c.imag], axis=1)

        return spec_dict

    def _compute_loss(self,
                      masked_stfts: Dict[str, torch.Tensor],
                      gt_stfts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes loss (see self._loss_func, default: L1)

        Parameters
        ----------
        masked_stfts: dict
            Dictionary of masked complex spectrograms (computed with estimated masks)

        gt_stfts: dict
            Dictionary of GT spectrograms (can be magnitude or complex)

        Returns
        -------
        torch.Tensor
            Loss value
        """

        losses = [self._loss_func(torch.abs(masked_stfts[target]),
                                  torch.abs(gt_stfts[target]))
                  for target in self._targets]

        return torch.sum(torch.stack(losses))

    def train(self,
              epoch: int):
        if self._model_cfg.audio.random_start_frame:
            self._train_dataset.generate_start_frames(seed=epoch)
        if self._model_cfg.audio.random_gain:
            self._train_dataset.generate_gains(seed=epoch+42)
        losses = AverageMeter()
        self._model.train()
        pbar = tqdm(self._train_sampler, disable=False, miniters=1)
        pbar.set_description('Training batch')
        for train_dict in pbar:
            X_c = self._stft(train_dict['mix'].to(self._device))

            # X_mag.shape = B, C, T, F
            X_c, _ = self._build_spec(X_c)

            # TODO: Check!
            X_c = torch.cat([X_c.real, X_c.imag], axis=1)
            self._optimizer.zero_grad()

            model_outputs = self._model(X_c)

            mask_dict = self._build_mask_dict(model_outputs=model_outputs)
            masked_stfts = self._build_masked_stfts(input_stft=X_c,
                                                    mask_dict=mask_dict)
            gt_stfts = self._build_gt_spec_dict(train_dict)
            loss = self._compute_loss(masked_stfts, gt_stfts)

            loss.backward()
            self._optimizer.step()
            losses.update(loss.item(), self._num_channels)
            pbar.set_postfix(loss='{:.3f}'.format(losses.avg))

        train_loss = losses.avg
        self._train_losses.append(train_loss)

        return train_loss

    def val(self):
        losses = AverageMeter()
        self._model.eval()

        with torch.no_grad():
            for val_dict in self._val_sampler:
                X_c = self._stft(val_dict['mix'].to(self._device)) # B, C, F, T
                X_c, _ = self._build_spec(X_c)  # B, C, T, F
                X_c = torch.cat([X_c.real, X_c.imag], axis=1)
                model_outputs = self._model(X_c)  # B, C, T, F
                mask_dict = self._build_mask_dict(model_outputs=model_outputs)
                masked_stfts = self._build_masked_stfts(input_stft=X_c,
                                                        mask_dict=mask_dict)
                gt_stfts = self._build_gt_spec_dict(val_dict)
                loss = self._compute_loss(masked_stfts, gt_stfts)
                losses.update(loss.item(), self._num_channels)

            val_loss = losses.avg
            self._scheduler.step(val_loss)
            self._val_losses.append(val_loss)

        return val_loss

    def save_params(self,
                    params: dict):
        with open(Path(self._output_dir, 'spl_cac.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

    def separate(self,
                 mix: torch.Tensor):
        """
        mix.shape = C, T
        """
        X_c_input = self._stft(mix.unsqueeze(0))
        num_samples, num_channels, num_bins, num_frames = X_c_input.shape
        X_c, _ = self._build_spec(X_c_input)
        X_c = torch.cat([X_c.real, X_c.imag], axis=1)
        model_outputs = self._model.forward(X_c)

        mask_dict = dict()

        for target in model_outputs:
            mask = model_outputs[target]
            reshaped_mask = torch.cat(torch.split(mask, 1, 0), dim=2)[..., :num_frames, :]
            mask_dict[target] = self._extend_mask(reshaped_mask).permute(0, 1, 3, 2)

        X_c_ = torch.cat([X_c_input.real, X_c_input.imag], axis=1)
        masked_stfts = self._build_masked_stfts(X_c_,
                                                mask_dict)

        # return model_outputs
        estimates_dict = dict()
        with torch.no_grad():
            for target in self._targets:
                # The first two channels are for the real part
                real = masked_stfts[target][:, :2, :, :]

                # The latter two channels for the imaginary
                imag = masked_stfts[target][:, 2:, :, :]
                complex_spect = torch.complex(real, imag)
                estimates_dict[target] = torch.istft(complex_spect.squeeze(0),
                                                     n_fft=self._window_size,
                                                     hop_length=self._model_cfg.stft.hop_size,
                                                     center=self._model_cfg.stft.center,
                                                     window=self._window,
                                                     normalized=False,
                                                     onesided=True,
                                                     length=mix.shape[-1])

        return model_outputs, mask_dict, masked_stfts, estimates_dict
