<table border="0">
  <tr>
    <td><img src="web_content/thumbnail.png" alt="image description" width="1000"></td>
    <td><h1>Source Separation of Piano Concertos</h1>
      This repository provides a pipeline for  decomposing piano concerto recordings into separate piano and orchestral tracks.      
      Our approach investigates open-source spectrogram- and waveform-based approaches as well as hybrid models operating in both spectrogram and waveform domains. 
<br> <br>
</td>
  </tr>
</table>


## Installation

We recommend to do this inside a conda or virtual environment (requiring at least Python 3.8). As an alternative, you may also create the environment ``pc-separation`` as provided by the file ``environment.yml`` (which includes the jupyter package to run the demo files):
```
conda env create -f environment.yml
```

## Separating piano concertos
The notebook [``Separator.ipynb``](https://github.com/yiitozer/pc-separation/blob/master/Separator.ipynb) showcases an exemplary application. It includes downloading [pretrained weights](https://drive.google.com/drive/folders/1-zcdkHWUcfehaTjoxp-eCjAjZevDGxSu) of UMX, SPL, DMC, and HDMC models and the test dataset [PCD](https://www.audiolabs-erlangen.de/resources/MIR/PCD) are also provided in the notebook.

For further information and to listen to audio examples, please visit our [demo website](https://audiolabs-erlangen.de/resources/MIR/2023-PianoConcertoSeparation).


## Unison data generation

## Training
To be continued :ghost:


## References

F. Stöter, S. Uhlich, A. Liutkus, and Y. Mitsufuji, [Open-Unmix – A reference implementation for music source separation](https://github.com/sigsep/open-unmix-pytorch), Journal of Open Source Software, vol. 4, no. 41, 2019.

R. Hennequin, A. Khlif, F. Voituret, and M. Moussallam, [Spleeter: a fast and efficient music source separation tool with pre-trained models](https://github.com/deezer/spleeter/tree/master), Journal of Open Source Software, vol. 5, no. 50, p. 2154, 2020, Deezer Research. 

A. Défossez, N. Usunier, L. Bottou, and F. R. Bach, [Music source separation in the waveform domain](https://github.com/facebookresearch/demucs), 2019. 

A. Défossez, [Hybrid spectrogram and waveform source separation](https://github.com/facebookresearch/demucs), in Proceedings of the ISMIR 2021 Workshop on Music Source Separation, Online, 2021.
