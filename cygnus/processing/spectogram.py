import torchaudio
import torchaudio.transforms as T

def spectogram_transformation(waveform, sr, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512, mel=True):
    """
    Convert an audio file to a spectrogram or mel spectrogram.

    Args:
        filepath (str): Path to the audio file.
        sample_rate (int): Target sample rate for the audio.
        n_mels (int): Number of Mel filterbanks (used if mel=True).
        n_fft (int): FFT window size.
        hop_length (int): Hop length between frames.
        mel (bool): If True, returns a Mel spectrogram; else returns a linear spectrogram.

    Returns:
        torch.Tensor: A tensor of shape [1, n_mels or freq_bins, time] representing the spectrogram in decibels.
    """
    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    if mel:
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    else:
        transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        )

    spec = transform(waveform)
    spec_db = torchaudio.functional.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0)
    return spec_db