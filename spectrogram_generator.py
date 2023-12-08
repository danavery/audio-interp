import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class SpectrogramGenerator:
    def __init__(self, sample_rate, n_mels=64, n_fft=512, hop_length=256):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec_transformer = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.amplitude_to_db_transformer = AmplitudeToDB()

    def generate_mel_spectrogram(self, audio, sample_rate):
        self.spec_transformer.sample_rate = sample_rate
        mel_spec = self.spec_transformer(audio).squeeze(0)
        mel_spec_db = self.amplitude_to_db_transformer(mel_spec)
        return mel_spec_db

    def normalize_spectrogram(self, spec):
        spectrogram = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        return spectrogram
