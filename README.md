### Source Separation of Piano Concertos
This repository addresses the novel and rarely considered source separation task of decomposing piano concerto recordings into separate piano and orchestral tracks. 
Being a genre written for a pianist typically accompanied by an ensemble or orchestra, piano concertos often involve an intricate interplay of the piano and the entire orchestra,
leading to high spectro–temporal correlations between the constituent instruments. 
Moreover, in the case of piano concertos, the lack of multi-track data for training constitutes another challenge in view of data-driven source separation approaches. 
As a basis for our work, we adapt existing deep learning (DL) techniques, mainly used for the separation of popular music recordings. 
In particular, we investigate spectrogram- and waveform-based approaches  as well as hybrid models operating in both spectrogram and waveform domains. 
As a main contribution, we introduce a musically motivated data augmentation approach for training based on artificially generated samples. 
Furthermore, we systematically investigate the effects of various augmentation techniques for DL-based models. 
For our experiments, we use [PCD](https://www.audiolabs-erlangen.de/resources/MIR/PCD), an open-source dataset of multi-track piano concerto recordings.
Our main findings demonstrate that the best source separation performance is achieved by a hybrid model when combining all augmentation techniques.


