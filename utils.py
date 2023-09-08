import IPython.display as ipd
import numpy as np
import pandas as pd
from typing import List
import yaml


class Config:
    def __init__(self,
                 config_dict: dict):
        self._config_dict = config_dict
        self._update_cfg()

    def _update_cfg(self):
        for k, v in self._config_dict.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)


def read_config_yaml(config_path: str) -> Config:
    with open(config_path) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f'config_path cannot be read!')

    return Config(config_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


def bandwidth_to_max_bin(rate: float,
                         n_fft: int,
                         bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count
    Assuming lapped transforms such as STFT
    Parameters
    ----------
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns
    -------
        np.ndarray: maximum frequency bin
    """
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def audio_player_list(signals: List,
                      rates: List,
                      width: int = 270,
                      height: int = 40,
                      columns: List = None,
                      column_align: str = 'center'):
    """Generate a list of HTML audio players tags for a given list of audio signals.

    Notebook: B/B_PythonAudio.ipynb

    Args:
        signals (list): List of audio signals
        rates (list): List of sample rates
        width (int): Width of player (either number or list) (Default value = 270)
        height (int): Height of player (either number or list) (Default value = 40)
        columns (list): Column headings (Default value = None)
        column_align (str): Left, center, right (Default value = 'center')
    """
    pd.set_option('display.max_colwidth', None)

    if isinstance(width, int):
        width = [width] * len(signals)
    if isinstance(height, int):
        height = [height] * len(signals)

    audio_list = []
    for cur_x, cur_Fs, cur_width, cur_height in zip(signals, rates, width, height):
        audio_html = ipd.Audio(data=cur_x, rate=cur_Fs)._repr_html_()
        audio_html = audio_html.replace('\n', '').strip()
        audio_html = audio_html.replace('<audio ', f'<audio style="width: {cur_width}px; height: {cur_height}px" ')
        audio_list.append([audio_html])

    df = pd.DataFrame(audio_list, index=columns).T
    table_html = df.to_html(escape=False, index=False, header=bool(columns))
    table_html = table_html.replace('<th>', f'<th style="text-align: {column_align}">')
    ipd.display(ipd.HTML(table_html))
