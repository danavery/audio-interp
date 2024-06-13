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

    def generate_mel_spectrogram(self, audio):
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
        logger.info(spec.shape)
        fig, ax = plt.subplots(figsize=(10, 4))
        _ = specshow(
            spec.numpy().T,
            sr=input_sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        return fig

    @staticmethod
    def ablate_spectrogram(
        spec, num_time_slices, time_slice_size, num_mel_slices=1
    ):
        """
        Ablates the spectrogram by splitting it into multiple time and Mel slices
        and replacing each segment with the mean of the dataset values.

        Parameters:
            spec (torch.Tensor): The spectrogram tensor.
            num_time_slices (int): Number of time slices to split the spectrogram into.
            time_slice_size (int): The size of each time slice.
            num_mel_slices (int): Number of Mel slices to split each time slice into (default is 1).

        Returns:
            torch.Tensor: A stacked tensor of the ablated spectrogram slices.
        """
        MEAN_VALUE = 0.4670  # mean value of the UrbanSound8K dataset

        num_mels = spec.shape[1]
        mel_slice_size = num_mels // num_mel_slices
        spec_variants = []

        for i in range(num_time_slices):
            start = i * time_slice_size
            end = start + time_slice_size
            for mel_slice_index in range(num_mel_slices):
                mel_start = mel_slice_index * mel_slice_size
                mel_end = mel_start + mel_slice_size
                filtered_spec = spec.clone()
                filtered_spec[start:end, mel_start:mel_end] = MEAN_VALUE
                spec_variants.append(torch.tensor(filtered_spec))
        spec_variants = torch.stack(spec_variants)
        return spec_variants

    @staticmethod
    def isolate_spectrogram_segment(
        spec,
        time_slice_size,
        most_valuable_time_slice_index,
        num_mel_bands,
        num_mel_slices,
        most_valuable_mel_slice_index,
    ):
        """
        Isolates a specific segment of the spectrogram by zeroing out the rest.

        Parameters:
            spec (torch.Tensor): The spectrogram tensor.
            time_slice_size (int): The size of the segment to keep along the time dimension.
            most_valuable_time_slice_index (int): The index of the most valuable time segment.
            num_mel_bands (int): The total number of Mel bands.
            num_mel_slices (int): The number of Mel slices.
            most_valuable_mel_slice_index (int): The index of the most valuable Mel slice.

        Returns:
            torch.Tensor: A PyTorch tensor of the isolated segment.
        """
        logger.info(spec.shape)
        start_time = most_valuable_time_slice_index * time_slice_size
        end_time = start_time + time_slice_size
        mel_portion_size = num_mel_bands // num_mel_slices
        start_mel = mel_portion_size * most_valuable_mel_slice_index
        end_mel = start_mel + mel_portion_size
        logger.info(f"{start_time=} {end_time=} {start_mel=} {end_mel=}")
        most_valuable_spec = spec.clone()
        most_valuable_spec[0:start_time, :] = 0.0
        most_valuable_spec[end_time:, :] = 0.0
        most_valuable_spec[:, 0:start_mel] = 0.0
        most_valuable_spec[:, end_mel:] = 0.0
        return most_valuable_spec
