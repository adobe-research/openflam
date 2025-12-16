"""
Sound Event Detection (SED) Inference Example of OpenFLAM
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""
import torch
import numpy as np
import librosa
import scipy
from pathlib import Path

import openflam
from openflam.module.plot_utils import plot_sed_heatmap


# Configuration
MODEL_SAMPLE_RATE = 48000  # The sample rate of the model
TARGET_SAMPLE_RATE = 32000  # The sample rate of the audio plotting and saving
DURATION = 10.0
EMB_RATE = 3.2  # Embedding rate in Hz (frames per second)
MEDIAN_FILTER = 3  # Median filter size for post-processing
NUM_FRAMES = int(DURATION * EMB_RATE)
# Change the audio path to the audio you want to test.
# In here we show the example of https://www.youtube.com/watch?v=tA1s65o_kYM.
AUDIO_PATH = "test_data/test_example.mp3"
AUDIO_START = 23
AUDIO_END = 33
OUTPUT_DIR = Path("sed_output")  # Directory to save output figures
# Define target sound events
TEXTS = [
    "man speaking",
    "man talking through a walkie-talkie",
    "music",
    "breathing sound",
    "ratcheting",
]

# Define negative class (sounds that shouldn't be in the audio)
NEGATIVE_CLASS = [
    "ratcheting",
]

def main():
    """Main function to run SED inference and plot results."""

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize OpenFLAM model
    print("Loading OpenFLAM model...")
    flam_wrapper = openflam.OpenFLAM(
        model_name="v1-base", default_ckpt_path="/tmp/openflam"
    )
    flam_wrapper.to("cuda")
    print("Model loaded successfully!")

    # Load and prepare audio
    print(f"Loading audio from: {AUDIO_PATH}")
    print(f"Time segment: {AUDIO_START}s - {AUDIO_END}s")
    audio, sr = librosa.load(AUDIO_PATH, sr=MODEL_SAMPLE_RATE)
    audio = audio[int(AUDIO_START * sr) : int(AUDIO_END * sr)]

    # Convert to tensor and move to device
    audio_tensor = torch.tensor(audio).unsqueeze(0).to("cuda")

    # Run inference
    print("\nRunning SED inference...")
    with torch.no_grad():
        # Get local similarity using the wrapper's built-in method
        # This uses the unbiased method (Eq. 9 in the paper)
        act_map_cross = (
            flam_wrapper.get_local_similarity(
                audio_tensor,
                TEXTS,
                method="unbiased",
                cross_product=True,
            )
            .cpu()
            .numpy()
        )

    print(f"Activation map shape: {act_map_cross.shape}")

    # Apply median filtering for smoother results
    print("Applying median filter...")
    act_map_filter = []
    for i in range(act_map_cross.shape[0]):
        act_map_filter.append(
            scipy.ndimage.median_filter(act_map_cross[i], (1, MEDIAN_FILTER))
        )
    act_map_filter = np.array(act_map_filter)

    # Prepare similarity dictionary for plotting
    similarity = {
        f"{TEXTS[i]}": act_map_filter[0][i] for i in range(len(TEXTS))
    }

    # Prepare audio for plotting (resample to 32kHz)
    audio_plot = librosa.resample(
        audio, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE
    )

    # Generate and save visualization
    output_path = OUTPUT_DIR / f"sed_heatmap_{AUDIO_START}s-{AUDIO_END}s.png"
    print(f"\nGenerating visualization and saving to: {output_path}")
    plot_sed_heatmap(
        audio_plot,
        TARGET_SAMPLE_RATE,
        post_similarity=similarity,
        duration=DURATION,
        negative_class=NEGATIVE_CLASS,
        figsize=(14, 8),
        save_path=output_path,
    )

    print(f"\nDone! Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
