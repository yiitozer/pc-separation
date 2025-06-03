import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.load import load_datasets
from .utils import Config, EarlyStopping

EPSILON = 1e-10


class Solver(object):
    def __init__(self,
                 model_type: str,
                 device: torch.device,
                 optimizer_func: torch.optim,
                 model_cfg: Config,
                 dataset_dir: str,
                 output_dir: str,
                 train_folder_name: str,
                 val_folder_name: str,
                 train: bool,
                 dataloader_kwargs: dict):

        self._model_type = model_type
        self._device = device
        self._optimizer_func = optimizer_func
        self._model_cfg = model_cfg
        self._dataset_dir = dataset_dir
        self._output_dir = output_dir
        self._train_folder_name = train_folder_name
        self._num_channels = 1 if model_cfg.audio.mono else 2

        if train:
            self._train_dataset, self._val_dataset = load_datasets(dataset_dir=dataset_dir,
                                                                   sr=model_cfg.audio.sample_rate,
                                                                   chunk_dur=model_cfg.audio.chunk_dur,
                                                                   silence_dur=model_cfg.audio.silence_dur,
                                                                   fade_dur=model_cfg.audio.fade_dur,
                                                                   mono=model_cfg.audio.mono,
                                                                   random_start_frame=model_cfg.audio.random_start_frame,
                                                                   random_gain=model_cfg.audio.random_gain,
                                                                   random_silence=model_cfg.audio.random_silence,
                                                                   train_folder_name=train_folder_name,
                                                                   val_folder_name=val_folder_name)

            self._train_sampler = DataLoader(self._train_dataset,
                                             batch_size=model_cfg.train.batch_size,
                                             shuffle=True,
                                             **dataloader_kwargs)

            self._val_sampler = DataLoader(self._val_dataset,
                                           batch_size=1,
                                           **dataloader_kwargs)

        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._train_losses = list()
        self._val_losses = list()

        self._checkpoint_filename = self._model_type
        # TODO: Check functionality!!!
        self.train_times = list()
        self.es = EarlyStopping(patience=model_cfg.train.patience)
        # TODO: For best_epoch, we need a solution!
        self.best_epoch = 0

    def train(self,
              epoch: int):
        raise NotImplementedError()

    def val(self):
        raise NotImplementedError()


    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def train_losses(self):
        return self._train_losses

    @property
    def val_losses(self):
        return self._val_losses

    def load_best_model(self,
                        checkpoint_dir: str):
        """Loads the best model, meaning the state_dict when the validation loss was the least.

        Parameters
        ----------
        checkpoint_dir: str
            Directory path of the checkpoints

        """
        model_path = os.path.join(checkpoint_dir, f'{self._model_type}_best.pth')
        # print(f'Loading the best {self._model_type} model from {model_path}.')
        self._model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 
                                                            f'{self._model_type}_best.pth',), 
                                                            map_location=self._device))

    def load_checkpoint(self,
                        model_dir: str):

        print(f'Loading the {self._model_type} checkpoint from {model_dir}')
        model_dir = Path(model_dir).expanduser()

        checkpoint_path = Path(model_dir, self._checkpoint_filename + ".chkpnt")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(checkpoint["state_dict"], strict=False)
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        # TODO: REMOVE THIS!!
        try:
            self._scheduler.load_state_dict(checkpoint["scheduler"])
        except:
            print('Scheduler not saved...')

    def load_params(self,
                    model_dir: str):

        with open(Path(model_dir, self._checkpoint_filename + ".json"), "r") as stream:
            results = json.load(stream)
        self._train_losses = results["train_loss_history"]
        self._val_losses = results["val_loss_history"]
        self.train_times = results["train_time_history"]
        self.best_epoch = results["best_epoch"]
        self.es.best = results["best_loss"]
        self.es.num_bad_epochs = results["num_bad_epochs"]

        return results

    def save_checkpoint(self,
                        epoch: int,
                        best_loss: float,
                        is_best: bool):
        # save full checkpoint including optimizer
        checkpoint_dict = {
            'epoch': epoch,
            'state_dict': self._model.state_dict(),
            'best_loss': best_loss,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict(),
        }

        torch.save(checkpoint_dict, os.path.join(self._output_dir, f'{self._checkpoint_filename}.chkpnt'))

        if is_best:
            # save just the weights
            torch.save(checkpoint_dict['state_dict'], Path(self._output_dir, f'{self._checkpoint_filename}_best.pth'))

    def save_params(self,
                    params: dict):
        with open(Path(self._output_dir, f'{self._checkpoint_filename}.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))
