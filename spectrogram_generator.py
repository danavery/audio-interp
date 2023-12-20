import logging

import matplotlib.pyplot as plt
import torch
from librosa.display import specshow
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


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

    @staticmethod
    def normalize_spectrogram(spec):
        spectrogram = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
        return spectrogram

    @staticmethod
    def plot_spectrogram(input_sr, spec, hop_length):
        plt.close()
        fig, ax = plt.subplots(figsize=(5, 2))
        _ = specshow(
            spec.numpy(),
            sr=input_sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        return fig

    @staticmethod
    def split_spectrogram(
        spec, num_time_slices, time_slice_size, sample_rate, num_mel_slices=1
    ):
        num_mels = spec.shape[1]
        mel_slice_size = num_mels // num_mel_slices
        spec_variants = []
        for i in range(num_time_slices):
            start = i * time_slice_size
            end = start + time_slice_size
            modified_spec = spec.clone()
            modified_spec[start:end, :] = 0.4670
            for mel_slice_index in range(num_mel_slices):
                mel_start = mel_slice_index * mel_slice_size
                mel_end = mel_start + mel_slice_size
                filtered_spec = modified_spec.clone()
                filtered_spec[:, mel_start:mel_end] = 0.4670
                spec_variants.append(torch.tensor(filtered_spec))
        spec_variants = torch.stack(spec_variants)
        return spec_variants

    @staticmethod
    def pad_spec(
        spec,
        segment_size,
        most_valuable_time,
        num_mel_bands,
        num_mel_portions,
        most_valuable_mel_index,
    ):
        start_time = most_valuable_time * segment_size
        end_time = start_time + segment_size
        mel_portion_size = num_mel_bands // num_mel_portions
        start_mel = mel_portion_size * most_valuable_mel_index
        end_mel = start_mel + mel_portion_size
        logger.info(f"{start_time=} {end_time=} {start_mel=} {end_mel=}")
        # most_valuable_spec = spec[start:end, :]
        most_valuable_spec = spec.clone()
        most_valuable_spec[0:start_time, :] = 0.0
        most_valuable_spec[end_time:, :] = 0.0
        most_valuable_spec[:, 0:start_mel] = 0.0
        most_valuable_spec[:, end_mel:] = 0.0
        return most_valuable_spec
