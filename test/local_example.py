"""
Local Example: Sound Event Detection (SED) with OpenFLAM
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""
from pathlib import Path

import librosa
import numpy as np
import scipy
import torch

import openflam
from openflam.module.plot_utils import plot_sed_heatmap

# Configuration
OUTPUT_DIR = Path("sed_output")  # Directory to save output figures

# Define target sound events
TEXTS = [
    "breaking bones",
    "metallic creak",
    "tennis ball",
    "troll scream",
    "female speaker",
]

# Define negative class (sounds that shouldn't be in the audio)
NEGATIVE_CLASS = [
    "female speaker"
]

SR = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

flam = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path="/tmp/openflam")
flam.to(DEVICE)

# Load and prepare audio
audio, sr = librosa.load("test/test_data/test_example.wav", sr=SR)
audio = audio[: int(10 * sr)]

# Convert to tensor and move to device
audio_tensor = torch.tensor(audio).unsqueeze(0).to(DEVICE)

# Run inference
with torch.no_grad():
    # Get local similarity using the wrapper's built-in method
    # This uses the unbiased method (Eq. 9 in the paper)
    act_map_cross = (
        flam.get_local_similarity(
            audio_tensor,
            TEXTS,
            method="unbiased",
            cross_product=True,
        )
        .cpu()
        .numpy()
    )

# Apply median filtering for smoother results
act_map_filter = []
for i in range(act_map_cross.shape[0]):
    act_map_filter.append(scipy.ndimage.median_filter(act_map_cross[i], (1, 3)))
act_map_filter = np.array(act_map_filter)

# Prepare similarity dictionary for plotting
similarity = {f"{TEXTS[i]}": act_map_filter[0][i] for i in range(len(TEXTS))}

# Prepare audio for plotting (resample to 32kHz)
target_sr = 32000
audio_plot = librosa.resample(audio, orig_sr=SR, target_sr=target_sr)

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Generate and save visualization
output_path = OUTPUT_DIR / "sed_heatmap.png"
plot_sed_heatmap(
    audio_plot,
    target_sr,
    post_similarity=similarity,
    duration=10.0,
    negative_class=NEGATIVE_CLASS,
    figsize=(14, 8),
    save_path=output_path,
)

print(f"Plot saved: {output_path}")
