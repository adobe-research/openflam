# OpenFLAM
<p align="center">
  <img src="./assets/FLAM_SLOGAN.png" alt="Framewise Language-Audio Modeling" width="75%"/>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2505.05335"><img src="https://img.shields.io/badge/arXiv-2505.05335-brightgreen.svg?style=flat-square"/></a>
  <a href="https://test.pypi.org/project/openflam"><img src="https://badge.fury.io/py/openflam"/></a>
</p>
 
### Joint Audio and Text Embeddings via Framewise Language-Audio Modeling (FLAM)

FLAM is a cutting-edge language–audio model that supports both zero-shot sound even detection and large-scale audio retrieval via free-form text.

This code accompanies the following ICML 2025 publication:
```
@inproceedings{flam2025,
  title = {FLAM: Frame-Wise Language-Audio Modeling},
  author = {Yusong Wu and Christos Tsirigotis and Ke Chen and Cheng-Zhi Anna Huang and Aaron Courville and Oriol Nieto and Prem Seetharaman and Justin Salamon},
  booktitle={International Conference on Machine Learning, ICML},
  year = {2025}
}
```

## Architecture

FLAM is based on contrastive language-audio pretraining, known as CLAP, and improve its capability by supporting the frame-wise event localization via learnable text and audio biases and scales.  
<p align="center">
  <img src="./assets/FLAM_ARCH.png" alt="FLAM Architecture" width="100%"/>
</p>

## Quick Start 

Install FLAM via PyPi:

```bash
pip install openflam
```

Two examples are provided:

1. [embedding_inference.py](./test/embedding_inference.py): to obtain audio and text embeddings and do sound event localization.
2. [sed_inference_and_plot.py](./test/sed_inference_and_plot.py) to do sound event localization and plot the results.

For the API documentation, please refer to [hook.py](./src/openflam/hook.py).

```python
import os

import librosa
import openflam
import torch


DEVICE = "cuda"  # cuda or cpu
SR = 48000       # Sampling Rate (FLAM requires 48kHz)

flam = openflam.OpenFLAM(
    model_name="v1-base", default_ckpt_path="/tmp/openflam"
).to(DEVICE)

# Sanity Check (Optional)
flam.sanity_check()

# load audio from 22-33 seconds
audio, sr = librosa.load("test_data/test_example.mp3", sr=SR)
audio = audio[int(23. * sr): int(33. * sr)]
audio_samples = torch.tensor(audio).unsqueeze(0).to(DEVICE) # [B, 480000 = 10 sec]

# Define text
text_samples = [
    "man speaking",
    "man talking through a walkie-talkie",
    "music",
    "breathing sound",
    "ratcheting"
]

# Get Global Audio Features (10sec = 0.1Hz embeddings)
audio_global_feature = flam.get_global_audio_features(
    audio_samples
)  # [B, 512]

# Get Local Audio Features (0.32sec = ~3Hz embeddings)
audio_local_feature = flam.get_local_audio_features(
    audio_samples
)  # [B, 32, 512] 32 is frame size (0.032 sec / frame)

# Get Text Features
text_feature = flam.get_text_features(text_samples)  # [B, 512]

# Get Local Similarity for Sound Event Detection
flamgram = flam.get_local_similarity(
    audio_samples,
    text_samples,
    method="unbiased",
    cross_product=True,
)
```

## License

Both **code** and **models** for OpenFLAM are released under an [Adobe Research License](./LICENSE). Please, review it carefully before using this technology.

## Pretrained Models

The pretrained checkpoints can be found [here](https://huggingface.co/kechenadobe/OpenFLAM/blob/main/open_flam_oct17.pth).

OpenFLAM automatically handles the downloading of the checkpoint. Please refer to the previous section for more details.

## Datasets

The original experimental results reported in [our paper](https://arxiv.org/abs/2505.05335) were obtained by the model trained on internal datasets that are not publicly shareable.

OpenFLAM is trained **on all publicly available datasets**, including: 

1. Datasets with coarse (aka, global or weak) labels: AudioSet-ACD (a LLM-based captioning for AudioSet), FreeSound, WavCaps, AudioCaps, Clotho;
2. Datasets with fine-grained (aka, local or strong) labels: AudioSet Strong, UrbanSED, DESED, Maestro, and Simulation data from AudioSet-ACD & FreeSound.

We report a comparison of the OpenFLAM performance to the original paper report (the global retrieval metrics --ie, A2T and T2A-- are R@1 / R@5):
<p align="center">
  <img src="./assets/Exp.png" alt="FLAM Exp" width="100%"/>
</p>


## Citation

If you use OpenFLAM, please cite our main work:

```
@inproceedings{flam2025,
  title = {FLAM: Frame-Wise Language-Audio Modeling},
  author = {Yusong Wu and Christos Tsirigotis and Ke Chen and Cheng-Zhi Anna Huang and Aaron Courville and Oriol Nieto and Prem Seetharaman and Justin Salamon},
  booktitle={International Conference on Machine Learning, ICML},
  year = {2025}
}
```

