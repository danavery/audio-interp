import numpy as np
import torch
from spectrogram_generator import SpectrogramGenerator
from torchaudio.transforms import Resample
import logging

logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel("INFO")


class AudioFileFeatureExtractor:
    def __init__(self, model_handler, dataset_handler):
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler

    def make_spec_plot(
        self,
        file_name=None,
        audio_class=None,
        feature_extractor=None,
        hop_length=None,
        selection_method="random",
    ):
        (
            spec,
            input_sr,
            audio,
            file_name,
            audio_class,
        ) = self._create_spec_data(
            file_name, audio_class, feature_extractor, selection_method, truncate=True
        )
        fig = SpectrogramGenerator.plot_spectrogram(input_sr, spec.transpose(0, 1), hop_length)
        return fig, audio[0].numpy(), file_name, audio_class, input_sr

    def _create_spec_data(
        self, file_name, audio_class, feature_extractor, selection_method, truncate
    ):
        file_name, audio_class, waveform, file_sr = self.dataset_handler.load_file(
            file_name, audio_class, selection_method
        )
        waveform = torch.unsqueeze(waveform, 0)
        if True:
            input_sr = feature_extractor.sampling_rate
            audio, spec = self.make_spec_from_ast(
                waveform, file_sr, input_sr, feature_extractor, truncate
            )
        # else:
        #     input_sr = 22050
        #     audio, spec = self._make_spec_from_local(waveform, file_sr, input_sr)
        #     hop_length = 256
        return spec, input_sr, audio, file_name, audio_class

    @staticmethod
    def _resample(audio, file_sr, input_sr):
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if file_sr != input_sr:
            resampler = Resample(file_sr, input_sr)
            audio = resampler(audio)
        else:
            logger.info("No resampling")
        num_samples = audio.shape[-1]
        total_duration = num_samples / input_sr

        return audio, num_samples, total_duration

    # def _make_spec_from_local(self, waveform, file_sr, input_sr):
    #     audio, _, duration = self._resample(waveform, file_sr, input_sr)
    #     spec_generator = SpectrogramGenerator(input_sr)
    #     spec = spec_generator.generate_mel_spectrogram(audio, input_sr)
    #     spec = spec_generator.normalize_spectrogram(spec)
    #     return audio, spec

    def make_spec_from_ast(
        self, waveform, file_sr, output_sr, feature_extractor, truncate=False
    ):
        # resampled_audio, _, duration = self._resample(waveform, file_sr, output_sr)
        resampled_audio = waveform
        inputs = feature_extractor(
            resampled_audio.numpy(),
            sampling_rate=output_sr,
            return_tensors="pt",
            padding="max_length",
        )
        spec = inputs["input_values"]
        spec = torch.squeeze(spec, 0)
        if truncate:
            actual_frames = np.ceil(len(resampled_audio[0]) / 160).astype(int)
            spec = spec[:actual_frames, :]
        return resampled_audio, spec
