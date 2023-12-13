from mmap_ninja.ragged import RaggedMmap
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Tuple


class PCDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 sr: int = 44100,
                 mono: bool = False,
                 split: str = 'train',
                 chunk_dur: float = 20.0,
                 silence_dur: float = 4.0,
                 fade_dur: float = 0.1,
                 random_start_frame: bool = True,
                 random_silence: bool = False,
                 random_gain: bool = True):

        self._dataset_dir = dataset_dir
        self._sr = sr
        self._mono = mono
        self._split = split
        self._chunk_dur = chunk_dur
        self._random_start_frame = random_start_frame
        self._random_silence = random_silence
        self._audio_files = RaggedMmap(os.path.join(dataset_dir, split))
        self._num_samples_chunk = int(chunk_dur * sr)

        # A list of values between 0 and 1, of size dataset
        # These values should be recomputed at the beginning of each epoch.
        # The idea here is to save runtime, since random number generation is a costly operation.
        self._start_frames = np.random.random(size=len(self._audio_files)) if random_start_frame else None
        self._gains = np.random.random(size=len(self._audio_files)) if random_gain else None
        self._random_silence_start_frames = np.random.random(size=len(self._audio_files)) if random_silence else None
        self._silence_win = self._create_silence_win(silence_dur, fade_dur) if random_silence else None
        self._silence_win_size = len(self._silence_win) if random_silence else None

    def _create_silence_win(self,
                            silence_dur,
                            fade_dur):
        num_samples_fade = int(fade_dur * self._sr)
        fadein = np.linspace(0.0, 1.0, num_samples_fade)
        fadeout = np.linspace(1.0, 0.0, num_samples_fade)
        num_samples_silence = int(silence_dur * self._sr)
        silence = np.zeros((num_samples_silence))

        return np.concatenate([fadeout, silence, fadein])

    def __getitem__(self,
                    index: int) -> dict:
        audio_files = self._audio_files[index]
        num_samples_audio = audio_files.shape[1]

        if self._random_start_frame:
            if num_samples_audio < self._num_samples_chunk:
                raise ValueError(f'Audio duration is smaller than the chunk duration for index {index}!')

            if self._start_frames is None:
                start_frame = torch.randint(0, num_samples_audio - self._num_samples_chunk, (1,))
            else:
                start_frame = int(self._start_frames[index] * (num_samples_audio - self._num_samples_chunk))

            segment = np.array(self._audio_files[index][:, start_frame:start_frame + self._num_samples_chunk, :])

        else:
            segment = np.array(self._audio_files[index])

        # TODO: This will be deprecated
        # If mix is already given in the dataset, don't apply random gains.
        if segment.shape[2] == 3:
            piano = segment[:, :, 0]
            orch = segment[:, :, 1]
            mix = segment[:, :, 2]

        else:
            if self._gains is not None:
                gain = self._gains[index]
            else:
                gain = 0.5

            if self._random_silence_start_frames is not None:
                silence_coeff = self._random_silence_start_frames[index]
                random_silence_start = int(np.floor(silence_coeff * (self._num_samples_chunk - self._silence_win_size)))

                # If the coefficient is smaller than 0.5, add a silent section to the piano part,
                # else to the orchestra
                if silence_coeff < 0.5:
                    segment[:, random_silence_start:random_silence_start+self._silence_win_size, 0] *= self._silence_win
                else:
                    segment[:, random_silence_start:random_silence_start+self._silence_win_size, 1] *= self._silence_win

            piano = segment[:, :, 0] * gain
            orch = segment[:, :, 1] * (1 - gain)
            mix = piano + orch

        return {'piano': piano,
                'orch': orch,
                'mix': mix}

    @property
    def random_start_frame(self):
        return self._random_start_frame

    @random_start_frame.setter
    def random_start_frame(self, rsf):
        self._random_start_frame = rsf

    def __len__(self) -> int:
        return len(self._audio_files)

    def generate_start_frames(self,
                              seed):
        np.random.seed(seed)
        self._start_frames = np.random.random(size=len(self._audio_files))

    def generate_gains(self,
                       seed):
        np.random.seed(seed)
        self._gains = np.random.random(size=len(self._audio_files))

    def generate_random_silence(self,
                                seed):
        np.random.seed(seed)
        self._random_silence_start_frames = np.random.random(size=len(self._audio_files))


def load_datasets(dataset_dir: str,
                  sr: int = 44100,
                  chunk_dur: float = 20.0,
                  silence_dur: float = 4.0,
                  fade_dur: float = 0.1,
                  mono: bool = False,
                  random_start_frame: bool = False,
                  random_silence: bool = False,
                  random_gain: bool = False,
                  train_folder_name: str = 'train',
                  val_folder_name: str = 'val',
                  seed: int = 42,
                  ) -> Tuple[PCDataset, PCDataset]:
    """Loads the specified dataset from commandline arguments
    Returns:
        train_dataset, validation_dataset
    """
    random.seed(seed)
    train_dataset = PCDataset(dataset_dir=dataset_dir,
                              split=train_folder_name,
                              sr=sr,
                              chunk_dur=chunk_dur,
                              silence_dur=silence_dur,
                              fade_dur=fade_dur,
                              mono=mono,
                              random_start_frame=random_start_frame,
                              random_silence=random_silence,
                              random_gain=random_gain)

    val_dataset = PCDataset(dataset_dir=dataset_dir,
                            split=val_folder_name,
                            sr=sr,
                            chunk_dur=chunk_dur,
                            mono=mono,
                            random_start_frame=False,
                            random_silence=False,
                            random_gain=False)

    return train_dataset, val_dataset
