import numpy as np
import torch
from scipy.signal import butter, lfilter


class MelBandFilter:
    def __init__(self, mel_bins, sample_rate):
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
        filtered_audio = MelBandFilter._bandpass_filter(
            audio_np, low_freq, high_freq, self.sample_rate, order=order
        )
        return torch.tensor(filtered_audio)

    def filter_time_slice(
        self, audio, mel_bin_range, num_time_bins, requested_time_bin
    ):
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

    def _bandpass_filter(data, lowcut, highcut, sample_rate, order=2):
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def _hz_to_mel(frequency, htk=False):
        if htk:
            return 2595 * np.log10(1 + frequency / 700)
        else:
            return 1127 * np.log(1 + frequency / 700)

    def _mel_to_hz(mel, htk=False):
        if htk:
            return 700 * (10 ** (mel / 2595) - 1)
        else:
            return 700 * (np.exp(mel / 1127) - 1)
