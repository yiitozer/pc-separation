import IPython.display as ipd
import numpy as np
import pandas as pd
from typing import List
import yaml

from model.umx import UMXSeparator
from solver.demucs import DemucsSolver
from solver.spl import SPLSolver
from solver.utils import Config


def read_config_yaml(config_path: str) -> Config:
    with open(config_path) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f'config_path cannot be read!')

    return Config(config_dict)


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

# For demonstration purposes
def init_separator(model_type, device='cuda'):
    if model_type == 'UMX06':
        model_dir = 'checkpoints/UMX06_R_H_HU_HUS'
        config_path = 'config/cfg_umx_piano.yaml'
        separator = UMXSeparator

    elif model_type == 'UMX20':
        model_dir = 'checkpoints/UMX20_R_H_HU_HUS'
        config_path = 'config/cfg_umx_piano_20sec.yaml'
        separator = UMXSeparator

    elif model_type == 'SPL':
        model_dir = 'checkpoints/SPL20_R_H_HU_HUS'
        config_path = 'config/cfg_spl.yaml'
        separator = SPLSolver

    elif model_type == 'DMC':
        model_dir = 'checkpoints/DMC20_R_H_HU_HUS'
        config_path = 'config/cfg_demucs.yaml'
        separator = DemucsSolver

    elif model_type == 'HDMC':
        model_dir = 'checkpoints/HDMC20_R_H_HU_HUS'
        config_path = 'config/cfg_hdemucs.yaml'
        separator = DemucsSolver

    else:
        raise ValueError('Wrong model_type. Please choose among UMX06, UMX20, SPL, DMC and HDMC')


    model_cfg = read_config_yaml(config_path)

    if 'UMX' in model_type:
        separator = separator(device=device,
                              model_cfg=model_cfg,
                              softmask=True,
                              residual=False)

        separator.load_model(model_dir=model_dir,
                             targets=['piano', 'orch'])

    else:
        separator = separator(device=device,
                              model_cfg=model_cfg,
                              train=False)

        separator.load_best_model(model_dir)

    return separator


