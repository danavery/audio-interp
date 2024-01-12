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

    def make_spec_from_dataset(
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
        ) = self._create_spec_data_from_filename(
            file_name, audio_class, feature_extractor, selection_method, truncate=True
        )
        logger.info(input_sr)
        fig = SpectrogramGenerator.plot_spectrogram(input_sr, spec, hop_length)
        logger.info(f"{audio.shape} {file_name}, {input_sr}")
        return fig, audio.numpy(), file_name, audio_class, input_sr

    def make_spec_from_input(
        self, feature_extractor, hop_length, truncate=True, audio=None
    ):
        sr, waveform = audio
        waveform = torch.tensor(
            waveform, dtype=torch.float
        )  # [samples, channels] or [samples] from Gradio Audio component
        # convert [samples] to [samples, channels]
        waveform = self._fix_waveform_channels(waveform)
        logger.info(waveform.shape)
        waveform, _, _ = self._resample(waveform, 16000)
        logger.info(waveform.shape)
        input_sr, audio, spec = self.make_spec(feature_extractor, truncate, waveform)
        fig = SpectrogramGenerator.plot_spectrogram(input_sr, spec, hop_length)
        return fig

    def _fix_waveform_channels(self, waveform):
        logger.info(f"{waveform.shape=}")
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(1)
        # then transpose it to [channels, samples]
        waveform = waveform.transpose(0, 1)
        return waveform

    def _create_spec_data_from_filename(
        self, file_name, audio_class, feature_extractor, selection_method, truncate
    ):
        file_name, audio_class, waveform, file_sr = self.dataset_handler.load_file(
            file_name, audio_class, selection_method
        )
        logger.info(waveform.shape)
        waveform = self._fix_waveform_channels(waveform)

        input_sr, audio, spec = self.make_spec(feature_extractor, truncate, waveform)
        return spec, input_sr, audio, file_name, audio_class

    def make_spec(self, feature_extractor, truncate, waveform):
        logger.info(waveform.shape)
        input_sr = feature_extractor.sampling_rate
        audio, spec = self.make_spec_with_ast_extractor(
            waveform, input_sr, feature_extractor, truncate
        )
        return input_sr, audio, spec

    @staticmethod
    def _resample(audio, file_sr, input_sr=16000):
        logger.info(f"{audio.shape=} {file_sr=}")
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if file_sr != input_sr:
            resampler = Resample(file_sr, input_sr)
            audio = resampler(audio)
        else:
            logger.info("No resampling")
        num_samples = audio.shape[1]
        total_duration = num_samples / input_sr
        logger.info(audio.shape)
        return audio, num_samples, total_duration

    @staticmethod
    def make_spec_with_ast_extractor(
        waveform, output_sr, feature_extractor, truncate=False, hop_length=160
    ):
        resampled_audio = waveform  # [1, samples]
        logger.info(resampled_audio.shape)
        inputs = feature_extractor(
            resampled_audio.numpy(),
            sampling_rate=output_sr,
            return_tensors="pt",
            padding="max_length",
        )
        spec = inputs["input_values"]  # [batch(1), frames, freq_bins]
        logger.info(spec.shape)
        spec = torch.squeeze(spec, 0)  # [frames, freq_bins]
        logger.info(spec.shape)
        if truncate:
            logger.info(f"{resampled_audio.shape[1]=} {hop_length=}")
            actual_frames = torch.ceil(
                torch.tensor(resampled_audio.shape[1] / hop_length)
            ).int()
            logger.info(actual_frames)
            spec = spec[:actual_frames, :]

        logger.info(f"{spec.shape=}")
        return resampled_audio, spec

    @staticmethod
    def make_spec_from_waveform(waveform, feature_extractor):
        input_sr = feature_extractor.sampling_rate
        waveform = torch.tensor(waveform[1])
        waveform = torch.unsqueeze(waveform, 0)
        raw_audio, spec = AudioFileFeatureExtractor.make_spec_with_ast_extractor(
            waveform, input_sr, feature_extractor, truncate=False
        )
        return input_sr, raw_audio, spec
