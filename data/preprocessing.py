from mmap_ninja.ragged import RaggedMmap
import numpy as np
import os
import torchaudio


class DataProcessor:
    def __init__(self,
                 dataset_dir='/mnt/Work/WorkYO/pc-pipeline/datasets/20221124_dataset_split/',
                 out_dir='.',
                 split='train',
                 mono=False,
                 target_sr=44100,
                 batch_size=32):
        self._dataset_dir = dataset_dir
        self._out_dir = out_dir
        self._mono = mono
        self._target_sr = target_sr
        self._split = split
        self._audio_filenames = os.listdir(os.path.join(dataset_dir, split))

        if not os.path.exists(os.path.join(out_dir, split)):
            os.makedirs(os.path.join(out_dir, split))
            RaggedMmap.from_generator(
                # Directory in which the memory map will be persisted
                out_dir=os.path.join(out_dir, split),

                # Something that yields np.ndarray
                sample_generator=map(self._audio_to_np, self._audio_filenames),

                # Maximum number of samples to keep in memory before flushing to disk
                batch_size=batch_size,

                # Show/hide progress bar
                verbose=True
            )
        else:
            print(os.path.join(out_dir, split) + ' already exists, Skipping!!')

    def _audio_to_np(self, audio_filename):
        piano, piano_sr = torchaudio.load(os.path.join(self._dataset_dir, self._split, audio_filename, 'piano.wav'))
        orch, orch_sr = torchaudio.load(os.path.join(self._dataset_dir, self._split, audio_filename, 'orchestra.wav'))
        mix, mix_sr = torchaudio.load(os.path.join(self._dataset_dir, self._split, audio_filename, 'mix.wav'))

        piano = self._resample_if_needed(piano, piano_sr)
        orch = self._resample_if_needed(orch, orch_sr)
        mix = self._resample_if_needed(mix, mix_sr)

        piano = self._reshape(piano)
        orch = self._reshape(orch)
        mix = self._reshape(mix)

        # Piano in index 0, Orchestra 1, Mix 2
        x = np.concatenate([piano, orch, mix], axis=2)

        print(x.shape)
        return np.concatenate([piano, orch, mix], axis=2)

    def _resample_if_needed(self, audio, sr):
        if self._target_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._target_sr)
            return resampler(audio)
        else:
            return audio

    def _reshape(self, audio):
        if self._mono:
            return audio.numpy().reshape(1, -1, 1)
        else:
            return audio.numpy().reshape(2, -1, 1)


