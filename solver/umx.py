import copy
import numpy as np
import os
import sklearn.preprocessing
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import Parameter, Sequential, ReLU


from dsp.transforms import TorchSTFT
from model.umx import ComplexNorm, OpenUnmix
from . import Solver
from utils import AverageMeter, bandwidth_to_max_bin, Config

EPSILON = 1e-10


class SpectralFlux(nn.Module):
    def __init__(self):
        super(SpectralFlux, self).__init__()

    def forward(self, X):
        Y = torch.log(1 + 10 * torch.abs(torch.Tensor(X)))
        Y_diff = torch.diff(Y, dim=3)
        Y_diff_act = ReLU(inplace=False)(Y_diff)
        nov = torch.sum(Y_diff_act, dim=2)
        nov_norm = nov / nov.max()

        return nov_norm


class UMXSolver(Solver):
    def __init__(self,
                 model_type: str = 'umx',
                 device: torch.device = torch.device('cuda'),
                 optimizer_func: torch.optim = torch.optim.Adam,
                 model_cfg: Config = None,
                 dataset_dir: str = '',
                 output_dir: str = '',
                 train_folder_name: str = 'train_aug',
                 val_folder_name: str = 'val_aug',
                 train: bool = True,
                 dataloader_kwargs: dict = {}):

        super(UMXSolver, self).__init__(model_type, device, optimizer_func, model_cfg, dataset_dir, output_dir,
                                        train_folder_name, val_folder_name, train, dataloader_kwargs)

        self._target = self._model_cfg.target

        self._stft = TorchSTFT(n_fft=self._model_cfg.stft.window_size,
                               n_hop=self._model_cfg.stft.hop_size,
                               window=Parameter(torch.hann_window(self._model_cfg.stft.window_size)),
                               center=self._model_cfg.stft.center)

        self._encoder = Sequential(self._stft,
                                   ComplexNorm(mono=self._model_cfg.audio.mono)).to(self._device)

        self._train_losses = list()
        self._val_losses = list()
        self._stats_dict = dict()
        self._stats_filepath = os.path.join(self._dataset_dir, self._train_folder_name + '_stats.npy')

        self._stats_dict = self._get_statistics()
        self._training_dataset_mean = self._stats_dict[self._target + '_mean']
        self._training_dataset_std = self._stats_dict[self._target + '_std']

        self._max_bin = bandwidth_to_max_bin(self._model_cfg.audio.sample_rate,
                                             self._model_cfg.stft.window_size,
                                             self._model_cfg.model.bandwidth)

        self._model = OpenUnmix(
            input_mean=self._training_dataset_mean,
            input_scale=self._training_dataset_std,
            nb_bins=self._model_cfg.stft.window_size // 2 + 1,
            nb_channels=self._num_channels,
            hidden_size=self._model_cfg.train.hidden_size,
            max_bin=self._max_bin
        ).to(self._device)

        self._optimizer = self._optimizer_func(self._model.parameters(),
                                               lr=model_cfg.train.lr,
                                               weight_decay=model_cfg.train.weight_decay)

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            factor=self._model_cfg.train.lr_decay_gamma,
            patience=self._model_cfg.train.lr_decay_patience,
            cooldown=10,
        )
        self._checkpoint_filename = self._target

    def _get_statistics(self) -> dict:
        # Send encoder to CPU
        if os.path.isfile(self._stats_filepath):
            self._stats_dict = np.load(self._stats_filepath, allow_pickle=True).item()

        else:
            print('Training dataset statistics are being computed...')
            encoder = copy.deepcopy(self._encoder).to('cpu')

            # Set random_start_frame False
            self._train_dataset.random_start_frame = False
            piano_scaler = sklearn.preprocessing.StandardScaler()
            orch_scaler = sklearn.preprocessing.StandardScaler()

            for dataset_dict in tqdm(self._train_dataset):
                # Each item in the PCDataset returns a dictionary involving piano, orch and mix
                piano, orch, _ = dataset_dict['piano'], dataset_dict['orch'], dataset_dict['mix']
                X = encoder(torch.from_numpy(piano)[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)
                Y = encoder(torch.from_numpy(orch)[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

                piano_scaler.partial_fit(X.detach().numpy().squeeze())
                orch_scaler.partial_fit(Y.detach().numpy().squeeze())

            # set inital input scaler values
            piano_mean = piano_scaler.mean_
            piano_std = np.maximum(piano_scaler.scale_, 1e-4 * np.max(piano_scaler.scale_))

            orch_mean = orch_scaler.mean_
            orch_std = np.maximum(orch_scaler.scale_, 1e-4 * np.max(orch_scaler.scale_))

            self._stats_dict = {'piano_mean': piano_mean,
                                'piano_std': piano_std,
                                'orch_mean': orch_mean,
                                'orch_std': orch_std}

            np.save(self._stats_filepath, self._stats_dict)

            # Set it back to its initial value
            self._random_start_frame = self._model_cfg.audio.random_start_frame

        return self._stats_dict

    def train(self,
              epoch):
        # generate new random_start_frames
        if self._model_cfg.audio.random_start_frame:
            self._train_dataset.generate_start_frames(seed=epoch)
        if self._model_cfg.audio.random_gain:
            self._train_dataset.generate_gains(seed=epoch+42)
        if self._model_cfg.audio.random_silence:
            self._train_dataset.generate_random_silence(seed=epoch+12)

        losses = AverageMeter()
        self._model.train()
        pbar = tqdm(self._train_sampler, disable=False, miniters=1)
        pbar.set_description('Training batch')
        for train_dict in pbar:
            x = train_dict['mix'].to(self._device)
            y = train_dict[self._target].to(self._device)
            self._optimizer.zero_grad()
            X = self._encoder(x)
            Y_hat = self._model(X)
            Y = self._encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            loss.backward()
            self._optimizer.step()
            losses.update(loss.item(), Y.size(1))
            pbar.set_postfix(loss='{:.3f}'.format(losses.avg))

        train_loss = losses.avg
        self._train_losses.append(train_loss)

        return train_loss

    def val(self):
        losses = AverageMeter()
        self._model.eval()

        with torch.no_grad():
            for val_dict in self._val_sampler:
                x = val_dict['mix'].to(self._device)
                y = val_dict[self._target].to(self._device)

                X = self._encoder(x)
                Y_hat = self._model(X)
                Y = self._encoder(y)
                loss = torch.nn.functional.mse_loss(Y_hat, Y)
                losses.update(loss.item(), Y.size(1))

            val_loss = losses.avg
            self._scheduler.step(val_loss)
            self._val_losses.append(val_loss)

        return val_loss

    #def save_checkpoint(self,
    #                    epoch: int,
    #                    best_loss: float,
    #                    is_best: bool):
    #    # save full checkpoint including optimizer
    #    checkpoint_dict = {
    #            'epoch': epoch,
    #            'state_dict': self._model.state_dict(),
    #            'best_loss': best_loss,
    #            'optimizer': self._optimizer.state_dict(),
    #            'scheduler': self._scheduler.state_dict(),
    #    }
#
    #    torch.save(checkpoint_dict, os.path.join(self._output_dir, self._target + f'.chkpnt'))
#
    #    if is_best:
    #        # save just the weights
    #        torch.save(checkpoint_dict['state_dict'], Path(self._output_dir, self._target + "_best.pth"))

    # def save_params(self,
    #                 params: dict):
    #     with open(Path(self._output_dir, self._target + '.json'), 'w') as outfile:
    #         outfile.write(json.dumps(params, indent=4, sort_keys=True))

