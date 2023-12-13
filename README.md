# Source Separation of Piano Concertos

This repository addresses the novel and rarely considered source separation task of decomposing piano concerto recordings into separate piano and orchestral tracks. 
Being a genre written for a pianist typically accompanied by an ensemble or orchestra, piano concertos often involve an intricate interplay of the piano and the entire orchestra,
leading to high spectro–temporal correlations between the constituent instruments. Moreover, in the case of piano concertos, the lack of multi-track data for training constitutes another challenge in view of data-driven source separation approaches. As a basis for our work, we adapt existing deep learning (DL) techniques, mainly used for the separation of popular music recordings. In particular, we investigate spectrogram- and waveform-based approaches  as well as hybrid models operating in both spectrogram and waveform domains. As a main contribution, we introduce a musically motivated data augmentation approach for training based on artificially generated samples. Furthermore, we systematically investigate the effects of various augmentation techniques for DL-based models. Our experiments use [PCD](https://www.audiolabs-erlangen.de/resources/MIR/PCD), an open-source dataset of multi-track piano concerto recordings. Our main findings demonstrate that the best source separation performance is achieved by a hybrid model when combining all augmentation techniques.


## Installation

We recommend to do this inside a conda or virtual environment (requiring at least Python 3.8). As an alternative, you may also create the environment ``pc-separation`` as provided by the file ``environment.yml`` (which includes the jupyter package to run the demo files):
```
conda env create -f environment.yml
```

## Separating piano concertos
The notebook [``Separator.ipynb``](https://github.com/yiitozer/pc-separation/blob/master/Separator.ipynb) showcases an exemplary application. It includes downloading [pretrained weights](https://drive.google.com/drive/folders/1-zcdkHWUcfehaTjoxp-eCjAjZevDGxSu) of UMX, SPL, DMC, and HDMC models and the test dataset [PCD](https://www.audiolabs-erlangen.de/resources/MIR/PCD) are also provided in the notebook.

For further information and to listen to audio examples, please visit our [demo website](https://audiolabs-erlangen.de/resources/MIR/2023-PianoConcertoSeparation).


## Training
To be continued :ghost:


## References

F. Stöter, S. Uhlich, A. Liutkus, and Y. Mitsufuji, [Open-Unmix – A reference implementation for music source separation](https://github.com/sigsep/open-unmix-pytorch), Journal of Open Source Software, vol. 4, no. 41, 2019.

R. Hennequin, A. Khlif, F. Voituret, and M. Moussallam, [Spleeter: a fast and efficient music source separation tool with pre-trained models](https://github.com/deezer/spleeter/tree/master), Journal of Open Source Software, vol. 5, no. 50, p. 2154, 2020, Deezer Research. 

A. Défossez, N. Usunier, L. Bottou, and F. R. Bach, [Music source separation in the waveform domain](https://github.com/facebookresearch/demucs), 2019. 

A. Défossez, [Hybrid spectrogram and waveform source separation](https://github.com/facebookresearch/demucs), in Proceedings of the ISMIR 2021 Workshop on Music Source Separation, Online, 2021.
