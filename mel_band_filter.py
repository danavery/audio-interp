import logging

import numpy as np
import torch
from scipy.signal import butter, lfilter

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class MelBandFilter:
    """
    A class for applying Mel scale bandpass filtering to audio data.

    This class provides methods to apply bandpass filtering to audio signals
    based on the Mel scale, which is a perceptual scale of pitches judged by
    listeners to be equal in distance from one another. The scale is useful
    in audio processing, particularly for speech and music analysis.

    Attributes:
    -----------
    mel_bins : int
        The number of Mel bins to be used in the filter.
    sample_rate : int
        The sampling rate of the audio signal in Hz.
    mel_bin_freq_ranges : list of tuples
        The frequency range (in Hz) for each Mel bin.

    Methods:
    --------
    filter(audio, mel_bin_range, order=2):
        Filters the given audio signal within the specified range of Mel bins.

    filter_time_slice(audio, mel_bin_range, num_time_bins, requested_time_bin):
        Filters a specific time slice of the audio signal within the specified Mel bin range.
    """

    def __init__(self, mel_bins, sample_rate=16000):
        self.mel_bins = mel_bins
        self.sample_rate = sample_rate
        self.mel_bin_freq_ranges = self._get_mel_bin_freq_ranges()

    def filter(self, audio, mel_bin_range, order=2):
        audio_np = audio.numpy()
        mel_bin_freq_ranges = self.mel_bin_freq_ranges
        low_freq, high_freq = (
            mel_bin_freq_ranges[mel_bin_range[0]][0],
            mel_bin_freq_ranges[mel_bin_range[1]][1],
        )
        low_freq = max(low_freq, 1.0)
        logger.info(mel_bin_range)
        logger.info(f"{low_freq}")
        logger.info(f"{high_freq}")
        filtered_audio = MelBandFilter._bandpass_filter(
            audio_np, low_freq, high_freq, self.sample_rate, order=order
        )
        return torch.tensor(filtered_audio)

    def filter_time_slice(
        self, audio, mel_bin_range, num_time_bins, requested_time_bin
    ):
        logger.info(num_time_bins)
        logger.info(f"{requested_time_bin=}")
        time_bin_start_points = torch.linspace(
            0, len(audio), num_time_bins + 1, dtype=torch.int32
        )
        audio_time_bin = audio[
            time_bin_start_points[requested_time_bin] : time_bin_start_points[
                requested_time_bin + 1
            ]
        ]
        mel_filtered_bin_audio = self.filter(audio_time_bin, mel_bin_range)
        return mel_filtered_bin_audio

    def _get_mel_bin_freq_ranges(self, htk=False):
        """Get the frequency range (in Hz) for each mel bin."""
        mel_points = np.linspace(
            MelBandFilter._hz_to_mel(0, htk=htk),
            MelBandFilter._hz_to_mel(self.sample_rate / 2, htk=htk),
            self.mel_bins + 2,
        )
        hz_points = MelBandFilter._mel_to_hz(mel_points, htk=htk)
        return [
            (hz_points[i - 1], hz_points[i + 1]) for i in range(1, len(hz_points) - 1)
        ]

    @staticmethod
    def _bandpass_filter(data, lowcut, highcut, sample_rate, order=2):
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def _hz_to_mel(frequency, htk=False):
        if htk:
            return 2595 * np.log10(1 + frequency / 700)
        else:
            return 1127 * np.log(1 + frequency / 700)

    @staticmethod
    def _mel_to_hz(mel, htk=False):
        if htk:
            return 700 * (10 ** (mel / 2595) - 1)
        else:
            return 700 * (np.exp(mel / 1127) - 1)
