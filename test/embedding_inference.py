"""
The embedding inference example of OpenFLAM
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""

import os
import torch
import librosa

import openflam


flam_wrapper = openflam.OpenFLAM(
    model_name="v1-base", default_ckpt_path="/tmp/openflam"
)

flam_wrapper.to("cuda")

# Sanity Check
flam_wrapper.sanity_check()

# load audio from 22-33 seconds
audio, sr = librosa.load("test_data/test_example.wav", sr=48000)
audio = audio[: int(10 * sr)]

# Convert to tensor and move to device
audio_samples = torch.tensor(audio).unsqueeze(0) # [B, 480000 = 10 sec]
text_samples = [
    "female speech, woman speaking",
    "mechanisms",
    "animal",
    "explosion"
]

# Get Audio and Text Embeddings
audio_global_feature = flam_wrapper.get_global_audio_features(
    audio_samples
)  # [B, 512]
audio_local_feature = flam_wrapper.get_local_audio_features(
    audio_samples
)  # [B, 32, 512] 32 is frame size (0.032 sec / frame)

text_feature = flam_wrapper.get_text_features(text_samples)  # [B, 512]

# Get Local Similarity for SED
act_map_cross = flam_wrapper.get_local_similarity(
    audio_samples,
    text_samples,
    method="unbiased",
    cross_product=True,
)


print("Audio Global Embedding Shape:", audio_global_feature.shape)
print("Audio Local Embedding Shape:", audio_local_feature.shape)
print("Text Embedding Shape:", text_feature.shape)
print("Local Similarity Map Shape:", act_map_cross.shape)