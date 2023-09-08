import argparse
import glob
import librosa
import os
import pretty_midi
import pandas as pd
import sox
from tqdm import tqdm
from typing import List, Tuple

DATASET_SCHEMA = {'composer':     pd.Series(dtype='str'),
                  'type':        pd.Series(dtype='str'),
                  'dataset':      pd.Series(dtype='str'),
                  'wav_filename': pd.Series(dtype='str'),
                  'start':        pd.Series(dtype='float'),
                  'end':          pd.Series(dtype='float'),
                  'duration':     pd.Series(dtype='float'),
                  'sample_rate':  pd.Series(dtype='float')}

DATASET_DIR = '/run/user/1001/gvfs/smb-share:server=lin2.audiolabs.uni-erlangen.de,share=groupmm/Work/WorkYO/datasets/'


def get_start_end_from_midi(midi_filepath: str) -> Tuple[float, float]:
    """Given the midi filepath, return the start and ending time points of a music recording."""

    fn = os.path.join(midi_filepath)
    midi_data = pretty_midi.PrettyMIDI(fn)
    midi_list = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

    df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
    return df['Start'].min(), df['End'].max()


def get_wav_filepaths(dataset_dir: str) -> List:
    wav_filepaths = list()
    for filepath in glob.iglob(f'{dataset_dir}/**/*.wav',
                               recursive=True):
        wav_filepaths.append(filepath)

    return wav_filepaths


def append_to_dataset(df_dataset: pd.DataFrame,
                      row_dict: dict) -> pd.DataFrame:
    df_dataset = pd.concat([df_dataset,
                            pd.DataFrame([row_dict])],
                           axis=0,
                           ignore_index=True)

    return df_dataset


def write_dataset_df(dataset_df: pd.DataFrame,
                     out_dir: str,
                     dataset_name: str,
                     delimiter: str = ';'):
    dataset_df.to_csv(os.path.join(out_dir, dataset_name + '.csv'), sep=delimiter, index=False)


def generate_maestro_df(dataset_dir: str,
                        dataset_type: str = 'piano',
                        dataset_name: str = 'maestro-v3.0.0') -> pd.DataFrame:
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    # Read the original metadata .csv file for MAESTRO V3.
    df_orig = pd.read_csv(os.path.join(dataset_dir, dataset_type, dataset_name, 'maestro-v3.0.0.csv'))
    tmp = df_orig['canonical_composer'].str.split(' ')
    df_orig['composer'] = tmp.str[-1]

    for idx, row in tqdm(df_orig.iterrows()):
        composer = row['composer']
        # MIDI filepaths are given in the 'midi_filename' column
        midi_filepath = os.path.join(dataset_dir, dataset_type, dataset_name, row['midi_filename'])

        # Get the starting and ending time points of each .wav file
        start, end = get_start_end_from_midi(midi_filepath)
        duration = row['duration']

        # wav filepaths are given under the audio_filename column
        wav_filepath = os.path.join(dataset_dir, dataset_type, dataset_name, row['audio_filename'])
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        # Get sample_rate using sox
        sample_rate = sox.file_info.sample_rate(wav_filepath)

        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': start,
                                                 'end': end,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})

    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_atepp_df(dataset_dir: str,
                      dataset_type: str = 'piano',
                      dataset_name: str = 'ATEPP') -> pd.DataFrame:
    df_dataset = pd.DataFrame(DATASET_SCHEMA)
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))

    for wav_filepath in tqdm(wav_filepaths):
        composer = wav_filepath.split('/')[1].split('_')[-1]
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        midi_filepath = wav_filepath.replace('.wav', '.mid')

        # Get the starting and ending time points of each .wav file
        start, end = get_start_end_from_midi(midi_filepath)
        duration = librosa.get_duration(filename=wav_filepath)

        # Get sample_rate using sox
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': start,
                                                 'end': end,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})

    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_musicnet_piano_df(dataset_dir: str,
                               dataset_type: str = 'piano',
                               dataset_name: str = 'MusicNet') -> pd.DataFrame:
    df_dataset = pd.DataFrame(DATASET_SCHEMA)
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))

    for wav_filepath in wav_filepaths:
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        composer = wav_filepath.split('/')[-1].split('_')[1]
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)

        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_musicnet_orchestra_df(dataset_dir: str,
                                   dataset_type: str = 'orchestra',
                                   dataset_name: str = 'MusicNet') -> pd.DataFrame:
    df_dataset = pd.DataFrame(DATASET_SCHEMA)
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        composer = wav_filepath.split('/')[-1].split('_')[1]
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)

        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_rwc_piano_df(dataset_dir: str,
                          dataset_type: str = 'piano',
                          dataset_name: str = 'RWC'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]
        composer = ''
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})

    df_dataset = df_dataset.sort_values('wav_filename')

    return df_dataset


def generate_rwc_orchestra_df(dataset_dir: str,
                              dataset_type: str = 'orchestra',
                              dataset_name: str = 'RWC'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]
        composer = ''
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})

    df_dataset = df_dataset.sort_values('wav_filename')

    return df_dataset


def generate_orchset_orchestra_df(dataset_dir: str,
                                  dataset_type: str = 'orchestra',
                                  dataset_name: str = 'OrchSet'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        composer = wav_filepath.split('/')[-1].split('-')[0]
        if composer == 'Rimski':
            composer = 'Korsakov'

        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_bach10_orchestra_df(dataset_dir: str,
                                 dataset_type: str = 'orchestra',
                                 dataset_name: str = 'Bach10'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]
        composer = 'Bach'
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_symph_idmt_orchestra_df(dataset_dir: str,
                                     dataset_type: str = 'orchestra',
                                     dataset_name: str = 'SymphonyDataset_IDMT'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]

        composer = wav_filename.split('/')[-1].split('_')[0]
        composer = composer[0].upper() + composer[1:]
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_phenicx_orchestra_df(dataset_dir: str,
                                  dataset_type: str = 'orchestra',
                                  dataset_name: str = 'PHENICX-Anechoic'):
    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, dataset_name))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]
        if wav_filename.split('/')[-2] == 'mixaudio_wav_22050_mono':
            composer = wav_filename.split('/')[-1].split('.wav')[0]
        else:
            composer = wav_filename.split('/')[-2]

        composer = composer[0].upper() + composer[1:]
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': -1,
                                                 'end': -1,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})
    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_df_violin_etudes_orig(dataset_dir: str,
                                   dataset_type: str = 'orchestra',
                                   dataset_name: str = 'ViolinEtudesOriginal'):

    def _get_start_end(wav_file):
        filename = wav_file.split('/')[-1].split('.wav')[0]
        csv_dir = wav_file.split(filename)[0].replace('original', 'resynth')
        csv_filepath = os.path.join(csv_dir, filename + '.RESYN.csv')
        df_start_end = pd.read_csv(csv_filepath)

        for idx, row in df_start_end.iterrows():
            if row[1] > 0:
                start = row[0]
                break
        for idx, row in df_start_end.iterrows():
            if row[1] > 0:
                end_idx = idx

        if idx == end_idx:
            end_idx = idx
        else:
            end_idx += 1
        end = df_start_end.iloc[end_idx][0]

        return start, end

    wav_filepaths = get_wav_filepaths(os.path.join(dataset_dir, dataset_type, 'ViolinEtudes/original'))
    df_dataset = pd.DataFrame(DATASET_SCHEMA)
    for wav_filepath in tqdm(wav_filepaths):
        wav_filename = wav_filepath.split(f'{dataset_type}/')[1]
        composer = wav_filename.split('/')[2].split('Vol')[0].split('Op')[0]
        duration = librosa.get_duration(filename=wav_filepath)
        sample_rate = sox.file_info.sample_rate(wav_filepath)
        start, end = _get_start_end(wav_filepath)
        df_dataset = append_to_dataset(df_dataset,
                                       row_dict={'composer': composer,
                                                 'dataset': dataset_name,
                                                 'type': dataset_type,
                                                 'wav_filename': wav_filename,
                                                 'start': start,
                                                 'end': end,
                                                 'duration': duration,
                                                 'sample_rate': sample_rate})

    # Sort by composer
    df_dataset = df_dataset.sort_values('composer')

    return df_dataset


def generate_df_violin_etudes_sync(dataset_dir: str,
                                   dataset_type: str = 'orchestra',
                                   dataset_name: str = 'ViolinEtudesSync'):
    df_orig = pd.read_csv('orchestra/ViolinEtudesOriginal.csv', delimiter=';')
    df_sync = pd.DataFrame(DATASET_SCHEMA)
    for idx, row in tqdm(df_orig.iterrows()):
        sync_filename = row['wav_filename'].replace('.wav', '_tar.wav').replace('/original/', '/sync/')
        # wav filepaths are given under the audio_filename column
        wav_filepath = os.path.join(dataset_dir, dataset_type, sync_filename)
        if not os.path.isfile(wav_filepath):
            continue

        composer = row['composer']

        # Get the starting and ending time points of each .wav file
        start = row['start']
        end = row['end']
        duration = row['duration']

        # Get sample_rate using sox
        sample_rate = sox.file_info.sample_rate(wav_filepath)

        df_sync = append_to_dataset(df_sync,
                                    row_dict={'composer': composer,
                                              'dataset': dataset_name,
                                              'type': dataset_type,
                                              'wav_filename': sync_filename,
                                              'start': start,
                                              'end': end,
                                              'duration': duration,
                                              'sample_rate': sample_rate})
    # Sort by composer
    df_sync = df_sync.sort_values('composer')

    return df_sync


def generate_urmp_orchestra_df(dataset_dir: str,
                               dataset_type: str = 'orchestra',
                               dataset_name: str = 'URMP'):
    def _get_start_end(txt_filepath):
        with open(txt_filepath, 'r') as f:
            for line in f.readlines():
                time, f0 = line.split('\t')
                f0 = float(f0.strip('\n'))
                if f0 > 0:
                    start = float(time)
                    break
            f.close()
        with open(txt_filepath, 'r') as f:
            for line in f.readlines():
                time, f0 = line.split('\t')
                f0 = float(f0.strip('\n'))
                if f0 > 0:
                    end = float(time)

        return start, end

    df_dataset = pd.DataFrame(DATASET_SCHEMA)

    sub_dir = f'{dataset_dir}/{dataset_type}/{dataset_name}/Dataset'

    for work_name in tqdm(os.listdir(sub_dir)):
        if os.path.isdir(f'{sub_dir}/{work_name}'):
            work_dir = f'{sub_dir}/{work_name}'
            filenames = os.listdir(work_dir)
            wav_files = list()
            for filename in filenames:
                if filename.endswith('.wav'):
                    wav_files.append(filename)
            mix_start = 1e7
            mix_end = 0
            for wav_file in wav_files:
                if wav_file.startswith('AuSep'):
                    txt_filepath = os.path.join(work_dir, wav_file.replace('AuSep', 'F0s').replace('wav', 'txt'))
                    start, end = _get_start_end(txt_filepath)
                    if start < mix_start:
                        mix_start = start
                    if end > mix_end:
                        mix_end = end

                    wav_filename = f'{dataset}/Dataset/{work_name}/{wav_file}'
                    duration = librosa.get_duration(filename=os.path.join(dataset_dir, dataset_type, wav_filename))
                    sample_rate = sox.file_info.sample_rate(os.path.join(dataset_dir, dataset_type, wav_filename))
                    df_dataset = append_to_dataset(df_dataset,
                                                   row_dict={'composer': 'Unknown',
                                                             'dataset': dataset_name,
                                                             'type': dataset_type,
                                                             'wav_filename': wav_filename,
                                                             'start': start,
                                                             'end': end,
                                                             'duration': duration,
                                                             'sample_rate': sample_rate})

            for wav_file in wav_files:
                if wav_file.startswith('AuMix'):
                    start, end = mix_start, mix_end
                    wav_filename = f'{dataset}/Dataset/{work_name}/{wav_file}'
                    duration = librosa.get_duration(filename=os.path.join(dataset_dir, dataset_type, wav_filename))
                    sample_rate = sox.file_info.sample_rate(os.path.join(dataset_dir, dataset_type, wav_filename))
                    df_dataset = append_to_dataset(df_dataset,
                                                   row_dict={'composer': 'Unknown',
                                                             'dataset': dataset_name,
                                                             'type': dataset_type,
                                                             'wav_filename': wav_filename,
                                                             'start': start,
                                                             'end': end,
                                                             'duration': duration,
                                                             'sample_rate': sample_rate})

    df_dataset = df_dataset.sort_values('wav_filename')

    return df_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter the name of the dataset to generate the csv file.')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-D', '--dataset-dir', type=str, default=None)
    args = parser.parse_args()
    dataset = args.dataset
    dataset_dir = args.dataset_dir
    print(f'Dataset {dataset}.csv is processing....')

    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    if dataset == 'ATEPP':
        df_atepp = generate_atepp_df(dataset_dir=dataset_dir)
        write_dataset_df(df_atepp, out_dir='piano/', dataset_name='ATEPP')

    elif dataset == 'Bach10':
        df_bach10 = generate_bach10_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_bach10, out_dir='orchestra/', dataset_name='Bach10')

    elif dataset == 'MAESTRO':
        df_maestro = generate_maestro_df(dataset_dir=dataset_dir)
        write_dataset_df(df_maestro, out_dir='piano/', dataset_name='MAESTRO')

    elif dataset == 'MusicNet':
        df_musicnet_piano = generate_musicnet_piano_df(dataset_dir=dataset_dir)
        write_dataset_df(df_musicnet_piano, out_dir='piano/', dataset_name='MusicNet')

        df_musicnet_orch = generate_musicnet_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_musicnet_orch, out_dir='orchestra/', dataset_name='MusicNet')

    elif dataset == 'OrchSet':
        df_orchset = generate_orchset_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_orchset, out_dir='orchestra/', dataset_name='OrchSet')

    elif dataset == 'PHENICX-Anechoic':
        df_phenicx = generate_phenicx_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_phenicx, out_dir='orchestra/', dataset_name='PHENICX-Anechoic')

    elif dataset == 'RWC':
        df_rwc = generate_rwc_piano_df(dataset_dir=dataset_dir)
        write_dataset_df(df_rwc, out_dir='piano/', dataset_name='RWC')

        df_rwc_orch = generate_rwc_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_rwc_orch, out_dir='orchestra/', dataset_name='RWC')

    elif dataset == 'ViolinEtudes':
        df_violin_etudes = generate_df_violin_etudes_orig(dataset_dir=dataset_dir)
        write_dataset_df(df_violin_etudes, out_dir='orchestra/', dataset_name='ViolinEtudesOriginal')

        df_violin_etudes_sync = generate_df_violin_etudes_sync(dataset_dir=dataset_dir)
        write_dataset_df(df_violin_etudes_sync, out_dir='orchestra/', dataset_name='ViolinEtudesSync')

    elif dataset == 'URMP':
        df_urmp = generate_urmp_orchestra_df(dataset_dir=dataset_dir)
        write_dataset_df(df_urmp, out_dir='orchestra/', dataset_name='URMP')

    elif dataset == 'piano' or dataset == 'orchestra':
        df = pd.DataFrame(DATASET_SCHEMA)
        for csv_filename in sorted(os.listdir(dataset)):
            df_next = pd.read_csv(os.path.join(dataset, csv_filename), delimiter=';')
            df = pd.concat([df, df_next], axis=0)
        df.to_csv(f'{dataset}.csv', sep=';', index=False)

    else:
        raise ValueError(f'{dataset} is not avalid dataset. Please choose among ATEPP, Bach10, MAESTRO, MusicNet, '
                         f'OrchSet, PHENICX-Anechoic, RWC, SymphonyDataset_IDMT, URMP, ViolinEtudes.')

# class Dataset:
#     def __init__(self,
#                  dataset_name: str,
#                  dataset_type: str,
#                  dataset_dir: str):
#         self.name = dataset_name
#         self.type = dataset_type
#         self.dir = dataset_dir
#         self.df = pd.DataFrame(DATASET_SCHEMA)
#
#     def process(self):
#         self.wav_filepaths = get_wav_filepaths(self.dir)
#
#     def write_dataset_df(self,
#                          out_dir: str,
#                          delimiter: str = ';'):
#         self.df.to_csv(os.path.join(out_dir, self.name + '.csv'),
#                        sep=delimiter, index=False)
#
#     def append_to_dataset(self,
#                           row_dict: dict):
#         self.df = pd.concat([self.df,
#                              pd.DataFrame([row_dict])],
#                             axis=0,
#                             ignore_index=True)

