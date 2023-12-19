import torch
import torchaudio
from mel_band_filter import MelBandFilter
from urban_sound_dataset_handler import UrbanSoundDatasetHandler


dataset_handler = UrbanSoundDatasetHandler(regenerate=False)
_, _, audio, sample_rate = dataset_handler.load_file(
    file_name="31325-3-1-0.wav", selection_method="filename"
)

mel_bins = 128  # Number of mel bins
mel_bin_range = (80, 100)
mel_band_filter = MelBandFilter(mel_bins, sample_rate)

# filter entire waveform by mel bin range frequencies
filtered_audio_tensor = mel_band_filter.filter(audio, mel_bin_range)
filtered_audio_tensor = torch.unsqueeze(filtered_audio_tensor, 0)
torchaudio.save("filtered_audio.wav", filtered_audio_tensor, sample_rate)

# filter specified slice of waveform by mel bin range frequencies
filtered_with_time_audio_tensor = mel_band_filter.filter_time_slice(
    audio, mel_bin_range, 10, 7
)
filtered_with_time_audio_tensor = torch.unsqueeze(filtered_with_time_audio_tensor, 0)
torchaudio.save("filtered_time_audio.wav", filtered_with_time_audio_tensor, sample_rate)
