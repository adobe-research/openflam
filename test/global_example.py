"""
Global Example: Audio-Text Similarity with OpenFLAM
---------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""

import librosa
import torch

import openflam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 48000  # Sampling Rate (FLAM requires 48kHz)

flam = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path="/tmp/openflam").to(
    DEVICE
)

# Sanity Check (Optional)
flam.sanity_check()

# load audio
audio, sr = librosa.load("test/test_data/test_example.wav", sr=SR)
audio = audio[: int(10 * sr)]
audio_samples = torch.tensor(audio).unsqueeze(0).to(DEVICE)  # [B, 480000 = 10 sec]

# Define text
text_samples = [
    "breaking bones",
    "mechanical beep",
    "whoosh short",
    "troll scream",
    "female speaker",
]

# Get Global Audio Features (10sec = 0.1Hz embeddings)
audio_global_feature = flam.get_global_audio_features(audio_samples)  # [B, 512]

# Get Text Features
text_feature = flam.get_text_features(text_samples)  # [B, 512]

# Calculate similarity (dot product)
global_similarities = (text_feature @ audio_global_feature.T).squeeze(1)

print("\nGlobal Cosine Similarities:")
for text, score in zip(text_samples, global_similarities):
    print(f"{text}: {score.item():.4f}")
