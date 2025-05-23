from typing import List
import matplotlib.pyplot as plt
import numpy as np

def prediction_visual(
    batch: List[str], 
    labels: List[List[int]], 
    predictions: List[List[float]], 
    titles: List[str]
):
    """
    Plot waveform, ground truth labels, and predicted speech probabilities for multiple audio files.

    Args:
        batch (List[tuple(data,sr)])
        labels (List[List[int]]): Binary label sequences matching audio length.
        predictions (List[List[float]]): Predicted probabilities matching audio length.
        titles (List[str]): Titles for each subplot.

    Returns:
        List[matplotlib.axes.Axes]: List of axes objects for each subplot.
    """
    n = len(batch)
    fig, axs = plt.subplots(nrows=n, figsize=(30, 2 * n), dpi=200, sharex=False)

    if n == 1:
        axs = [axs]  # Ensure iterable

    for i in range(n):
        sr, data = batch[i]
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data / np.max(np.abs(data))
        time = np.arange(len(data)) / sr

        ax1 = axs[i]
        ax1.plot(time, data, color='grey')
        ax1.set_ylabel("Signal")
        ax1.set_yticks([-0.5, 0, 0.5])
        ax1.set_xlim(time[0], time[-1])
        ax1.set_ylim(-1.1, 1.1)

        ax2 = ax1.twinx()
        ax2.plot(time, labels[i], color='red', label='Label')
        ax2.plot(time, predictions[i], color='green', linestyle='--', label='Speech Probability')
        ax2.set_ylabel("Probability")
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_ylim(-0.1, 1.1)

        lines1, labels1_ = ax1.get_legend_handles_labels()
        lines2, labels2_ = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1_ + labels2_, loc='upper right')

        ax1.set_title(titles[i])
        if i == n - 1:
            ax1.set_xlabel("Time (s)")

    plt.tight_layout()
    return axs
