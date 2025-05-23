import torch
import torchaudio
import torchaudio.transforms as T

def stft_transformation(waveform, sr, sample_rate=16000, n_fft=127, hop_length=512):
    """
    Convert an audio file to a Short-Time Fourier Transform (STFT) spectrogram in dB.

    Args:
        filepath (str): Path to the audio file.
        sample_rate (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length between frames.

    Returns:
        torch.Tensor: A tensor of shape [1, freq_bins, time] representing the STFT magnitude in decibels.
    """
    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    magnitude = stft.abs().unsqueeze(0)
    magnitude_db = torchaudio.functional.amplitude_to_DB(magnitude, multiplier=10.0, amin=1e-10, db_multiplier=0)
    return magnitude_db
