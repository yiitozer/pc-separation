from setuptools import setup, find_packages
with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(name='pc-separation',
      version='1.0.0',
      author='Yigitcan Özer',
      author_email='yigitcan.oezer@audiolabs-erlangen.de',
      url='',
      download_url='',
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Intended Audience :: Developers",
          "Topic :: Multimedia :: Sound/Audio :: Analysis",
          "Programming Language :: Python :: 3",
      ],
      keywords='audio source separation',
      license='MIT',
      install_requires=['ipython >= 7.8.0, < 8.0.0',
                        'librosa >= 0.8.0, < 1.0.0',
                        'matplotlib >= 3.1.0, < 4.0.0',
                        'music21 >= 5.7.0, < 6.0.0',
                        'numba >= 0.51.0, < 0.55.0',
                        'numpy >= 1.17.0, < 2.0.0',
                        'pandas >= 1.0.0, < 2.0.0',
                        'pretty_midi >= 0.2.0, < 1.0.0',
                        'pysoundfile >= 0.9.0, < 1.0.0',
                        'scipy >= 1.7.0, < 2.0.0',
                        'libfmp >= 1.2.0, < 2.0.0',
                        'synctoolbox >= 1.3.0',
                        'sox == 1.4.1',
                        'tqdm >= 4.0.0',
                        'pyroomacoustics == 0.7.2',
                        'torch==1.13.0',
                        'torchaudio==0.13.0',
                        'torchvision==0.14.0'],
      python_requires='>=3.8, <4.0'
)


