"""
Plotting Utilities
--------------------------------------------------------
Paper: https://arxiv.org/abs/2505.05335
Code Maintainers: Ke Chen, Yusong Wu, Oriol Nieto, Prem Seetharaman
Support: Adobe Research
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display


plt.rcParams["font.family"] = "Times New Roman"


def plot_spec(wav, sr, title="", vmin=-8, vmax=1, save_path=None):
    """Plot and save the spectrogram.

    Args:
        wav (np.ndarray): Audio waveform array.
        sr (int): Sample rate of the audio.
        title (str): Title for the spectrogram plot. Defaults to "".
        vmin (float): Minimum value for the color scale. Defaults to -8.
        vmax (float): Maximum value for the color scale. Defaults to 1.
        save_path (str or Path): Path to save the spectrogram figure. If None, figure is not saved.
    """
    spec = np.log(np.abs(librosa.stft(wav, n_fft=512 + 256)))
    librosa.display.specshow(spec, sr=sr, vmin=vmin, vmax=vmax, cmap="magma")
    plt.title(title, fontsize=14)
    if save_path:
        plt.savefig(save_path)
        plt.close()


def plot_sed_heatmap(
    audio,
    sr,
    post_similarity=None,
    label=None,
    duration=10.0,
    title_fontsize=14,  # 1) Font size of the subplot titles
    label_fontsize=15,  # 2) Font size of the y-axis labels
    axis_fontsize=10,  # 3) Font size of the x-axis labels
    linewidth=0.8,  # 4) Width of the horizontal white lines separating rows
    negative_class=[],
    figsize=(14, 8),
    save_path=None,  # Path to save the figure (if provided, will save instead of show)
):
    """Plot multi-panel visualization with audio spectrogram and heatmaps.
        Plots:
        1) Audio spectrogram
        2) Post-processed activation heatmap
        3) Label heatmap (optional)

    Args:
        audio (np.ndarray): Audio waveform array.
        sr (int): Sample rate of the audio.
        post_similarity (dict): Dictionary mapping event labels to activation arrays.
            Each array should have shape (num_frames,).
        label (dict): Optional dictionary mapping event labels to ground truth label arrays.
            Each array should have shape (num_frames,). Defaults to None.
        duration (float): Duration of the audio segment in seconds. Defaults to 10.0.
        title_fontsize (int): Font size of the subplot titles. Defaults to 14.
        label_fontsize (int): Font size of the y-axis tick labels. Defaults to 15.
        axis_fontsize (int): Font size of the x-axis labels. Defaults to 10.
        linewidth (float): Width of the horizontal white lines separating rows. Defaults to 0.8.
        negative_class (list): List of event labels that are negative examples (will be colored red). Defaults to [].
        figsize (tuple): Figure size in inches (width, height). Defaults to (14, 8).
        save_path (str or Path): Path to save the figure. If provided, saves instead of showing. Defaults to None.
    """

    negative_class_num = len(negative_class)

    if label is not None:
        nrows = 3
    else:
        nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, dpi=300)
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 4))

    # 1) Plot the audio spectrogram in the top subplot
    plt.sca(axes[0])
    plot_spec(
        audio, sr, title="Audio spectrogram"
    )  # <-- your existing spectrogram plot function

    # Create our green→red colormap
    green_red_cmap = LinearSegmentedColormap.from_list(
        "GreenYellowRed", ["green", "yellow", "#ba001e"]
    )
    white_red_cmap = LinearSegmentedColormap.from_list(
        "WhiteYellowRed", ["white", "yellow", "#ba001e"]
    )

    # Helper to stack dict data into a matrix
    def dict_to_matrix(data_dict):
        labels_list = list(data_dict.keys())
        stacked = np.vstack([data_dict[lbl] for lbl in labels_list])
        return stacked, labels_list

    # Helper to plot a single heatmap (but do not add colorbar yet)
    def plot_heatmap(ax, data_dict, title, duration, cmap, negative_class_num):
        data_matrix, labels_list = dict_to_matrix(data_dict)
        n_labels, n_frames = data_matrix.shape

        # viridis, cividis, magma, inferno
        im = ax.imshow(
            data_matrix,
            aspect="auto",
            origin="upper",
            extent=[0, duration, 0, n_labels],
            cmap=cmap,  # Use our green→red colormap
            vmin=0,
            vmax=1,  # Fix range at [0..1]
        )

        # Draw horizontal white lines between each row to separate them visually
        for i in range(1, n_labels):
            ax.axhline(i, color="white", linewidth=linewidth)

        # Set fontsizes and labels
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_yticks(np.arange(n_labels) + 0.5)
        ax.set_yticklabels(labels_list[::-1], fontsize=label_fontsize)
        ax.set_xlabel("Time (s)", fontsize=axis_fontsize)
        yticklabels = ax.get_yticklabels()
        # make the last negative_class_num labels red
        for i in range(0, negative_class_num):
            yticklabels[i].set_color("red")

        return im

    im = plot_heatmap(
        axes[1],
        post_similarity,
        "FLAM Output",
        duration,
        cmap="viridis",
        negative_class_num=negative_class_num,
    )

    # 4) Label heatmap
    if label is not None:
        im = plot_heatmap(
            axes[2],
            label,
            "Label",
            duration,
            cmap="viridis",
            negative_class_num=negative_class_num,
        )

    # Create a single colorbar for all heatmaps on the right side
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes(
        [0.88, 0.14, 0.015, 0.32]
    )  # [left, bottom, width, height]
    if im is not None:
        fig.colorbar(im, cax=cbar_ax)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
