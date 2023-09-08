import argparse
from audiomentations import Compose, TimeStretch, PitchShift, Shift, RoomSimulator
import datetime
from functools import partial
import hashlib
import librosa
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import soundfile as sf
from tqdm.contrib.concurrent import process_map

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    if len(audio.shape) != 2:
        return np.stack([audio, audio])
    else:
        return audio


def get_info(item: pd.Series,
             split: str,
             filename_hash: str) -> str:
    txt = f'piano_composer:\t\t{item[0]}\npiano_wav_filename:\t\t{item[1]}\n'
    txt += f'piano_start:\t\t\t{item[2]}\norchestra_composer:\t\t{item[3]}\n'
    txt += f'orchestra_wav_filename:\t{item[4]}\norchestra_start:\t\t{item[5]}\n'
    txt += f'split\t\t\t\t{split}\nfilename:\t\t\t{filename_hash}'

    return txt


def _write_raw_audio_files(item: pd.Series,
                           output_dir: str,
                           split: str,
                           dataset_dir: str,
                           Fs: int = 44100,
                           mono: bool = False,
                           split_duration: float = 20.0):

    piano_wav_filename = item[1]
    piano_wav_filepath = os.path.join(dataset_dir, 'piano', piano_wav_filename)
    piano_start = item[2]
    orch_wav_filename = item[4]
    orch_wav_filepath = os.path.join(dataset_dir, 'orchestra', orch_wav_filename)
    orch_start = item[5]

    filename = piano_wav_filename + str(piano_start) + orch_wav_filename + str(orch_start)
    filename_hash = hashlib.md5(filename.encode()).hexdigest()  #

    if os.path.isdir(os.path.join(output_dir, split, filename_hash)):
        logging.info(f'Match piano {piano_wav_filepath} start: {piano_start}, '
                     f'orchestra {orch_wav_filepath} start: {orch_start} '
                     f'filename_hash: {filename_hash} already exists. Skipping.\n')
        return

    Path(os.path.join(output_dir, split, filename_hash)).mkdir(parents=True, exist_ok=True)
    logging.info(f'Matching piano {piano_wav_filepath} start: {piano_start}, '
                 f'orchestra {orch_wav_filepath} start: {orch_start} '
                 f'filename_hash: {filename_hash}.\n')
    try:
        piano, _ = librosa.load(piano_wav_filepath, sr=Fs, mono=mono, offset=piano_start, duration=split_duration)
        orch, _ = librosa.load(orch_wav_filepath, sr=Fs, mono=mono, offset=orch_start, duration=split_duration)

    except:
        logging.error(f'Error while matching piano {piano_wav_filepath} start: {piano_start}, '
                      f'orchestra {orch_wav_filepath} start: {orch_start}')
        return

    piano = mono_to_stereo(piano)
    orch = mono_to_stereo(orch)

    piano = librosa.util.normalize(piano, axis=1)
    orch = librosa.util.normalize(orch, axis=1)

    sf.write(os.path.join(output_dir, split, filename_hash, 'piano.wav'), piano.T, Fs)
    sf.write(os.path.join(output_dir, split, filename_hash, 'orchestra.wav'), orch.T, Fs)

    with open(os.path.join(output_dir, split, filename_hash, 'info.txt'), "w") as output:
        output.write(get_info(item, split, filename_hash))

INPUT_DIR = '/run/user/1001/gvfs/smb-share:server=lin2.audiolabs.uni-erlangen.de,share=groupmm/Work/WorkYO/pc-pipeline/datasets/20221124_dataset_split/train'


ROOM_AUGMENT = Compose([RoomSimulator(min_size_x=3, max_size_x=10,
                        min_size_y=3, max_size_y=10,
                        min_size_z=2.5, max_size_z=4,
                        max_order=7,
                        p=0.5)])


STEM_AUGMENT = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=0.1),
                        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2)])


def _augment_audiofiles(filename_hash : str,
                        output_dir: str,
                        input_dir: str = INPUT_DIR,
                        Fs: int = 44100,
                        # N=4096,
                        # H=1028,
                        split_duration: float = 20.0,
                        mono: bool = False,
                        room_augment: Compose = ROOM_AUGMENT,
                        stem_augment: Compose = STEM_AUGMENT):
    if os.path.isdir(os.path.join(output_dir, filename_hash)):
        logging.info(f'{os.path.join(output_dir, filename_hash)} already exists.\n')

        return

    Path(os.path.join(output_dir, filename_hash)).mkdir(parents=True, exist_ok=True)

    try:
        piano, _ = librosa.load(os.path.join(input_dir, filename_hash, 'piano.wav'), sr=Fs, mono=mono)
        orch, _ = librosa.load(os.path.join(input_dir, filename_hash, 'orchestra.wav'), sr=Fs, mono=mono)
        logging.info(f'Augmenting {filename_hash}.\n')
    except:
        logging.error(f'Files cannot be read for {filename_hash}')
        return

    augmented_piano = stem_augment(samples=piano, sample_rate=Fs)
    augmented_orch = stem_augment(samples=orch, sample_rate=Fs)

    augmented_piano = room_augment(samples=augmented_piano, sample_rate=Fs)
    room_augment.freeze_parameters()
    augmented_orch = room_augment(samples=augmented_orch, sample_rate=Fs)
    room_augment.unfreeze_parameters()

    augmented_piano = librosa.util.normalize(augmented_piano, axis=1)
    augmented_orch = librosa.util.normalize(augmented_orch, axis=1)

    gain = np.random.random_sample() * 0.8 + 0.1

    augmented_piano = augmented_piano[:int(split_duration * Fs), :] * gain
    augmented_orch = augmented_orch[:int(split_duration * Fs), :] * (1 - gain)

    # Random gain between 0.1 and 0.9
    mix = augmented_piano * gain + augmented_orch * (1 - gain)

    # Write .wav files
    try:
        logging.info(f'Writing {filename_hash}.\n')
        sf.write(os.path.join(output_dir, filename_hash, 'piano.wav'), augmented_piano.T, Fs)
        sf.write(os.path.join(output_dir, filename_hash, 'orchestra.wav'), augmented_orch.T, Fs)
        sf.write(os.path.join(output_dir, filename_hash, 'mix.wav'), mix.T, Fs)
    except:
        logging.error(f'An error occured while writing files for {filename_hash}.\n')

    with open(os.path.join(input_dir, filename_hash, 'info.txt'), 'r') as f:
        info = f.read()

    info += f'\npiano gain:\t\t\t{gain}'

    with open(os.path.join(output_dir, filename_hash, 'info.txt'), "w") as output:
        output.write(info)

    logging.info(info + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some text here ma love.')
    parser.add_argument('-s', '--split', type=str)
    parser.add_argument('-D', '--dataset-dir', type=str, default='')
    parser.add_argument('-a', '--audio_type', type=str, default='raw')
    parser.add_argument('-f', '--Fs', type=int, default=44100)
    parser.add_argument('-o', '--output_dir', type=str)

    args = parser.parse_args()
    split = args.split
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    audio_type = args.audio_type
    Fs = args.Fs

    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(f'Invalid dataset_dir: {dataset_dir}!')

    if split == 'val':
        df = pd.read_csv('val_match.csv', sep=';')
    elif split == 'train':
        df = pd.read_csv('train_match.csv', sep=';')
    else:
        raise ValueError(f'Invalid split: {split} ! Please choose among val and train.')

    print(f'Processing... Number of CPUs: {mp.cpu_count()}.')
    logging.basicConfig(filename=f'{datetime.datetime.now()}_{split}_{audio_type}.log',
                        level=logging.DEBUG)

    if audio_type == 'raw':
        with mp.Pool(processes=mp.cpu_count()) as pool:
            process_map(partial(_write_raw_audio_files,
                                dataset_dir=dataset_dir,
                                output_dir=output_dir,
                                split=split,
                                Fs=Fs),
                        df.values.tolist(),
                        max_workers=mp.cpu_count(),
                        chunksize=5)

    elif audio_type == 'augment':
        with mp.Pool(processes=mp.cpu_count()) as pool:
            process_map(partial(_augment_audiofiles, input_dir=dataset_dir, output_dir=output_dir, Fs=Fs),
                        os.listdir(dataset_dir),
                        max_workers=mp.cpu_count(),
                        chunksize=5)
    else:
        raise ValueError(f'Invalid audio_type: {audio_type}! Please choose among raw and augment.')
