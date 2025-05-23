import torchaudio.transforms as T

def mfcc_transformation(waveform, sr, sample_rate=16000, n_mfcc=64, n_mels=64, n_fft=1024, hop_length=512):
    """
    Convert an audio file to MFCC features.

    Args:
        filepath (str): Path to the audio file.
        sample_rate (int): Target sample rate for the audio.
        n_mfcc (int): Number of MFCC coefficients to return.
        n_mels (int): Number of Mel filterbanks.
        n_fft (int): FFT window size.
        hop_length (int): Hop length between frames.

    Returns:
        torch.Tensor: A tensor of shape [1, n_mfcc, time] representing the MFCC features.
    """
    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'mel_scale': 'htk'
        }
    )

    mfcc = mfcc_transform(waveform)
    return mfcc
