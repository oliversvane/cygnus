
import torchaudio.transforms as T

def raw_waveform_transformation(waveform, sr, sample_rate=16000):
    """
    Load an audio file and resample to the target sample rate.

    Args:
        filepath (str): Path to the audio file.
        sample_rate (int): Desired sample rate.

    Returns:
        torch.Tensor: A tensor of shape [1, num_samples] containing the waveform.
    """
    
    if sr != sample_rate:
        waveform = T.Resample(sr, sample_rate)(waveform)
    return waveform