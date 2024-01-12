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
            waveform,
            file_name,
            audio_class,
        ) = self._create_spec_data_from_filename(
            file_name, audio_class, feature_extractor, selection_method, truncate=True
        )
        sample_rate = feature_extractor.sampling_rate
        logger.info(sample_rate)
        fig = SpectrogramGenerator.plot_spectrogram(sample_rate, spec, hop_length)
        logger.info(f"{waveform.shape} {file_name}, {sample_rate}")
        return fig, waveform.numpy(), file_name, audio_class, sample_rate

    def make_spec_from_input(
        self, feature_extractor, hop_length, truncate=True, audio=None
    ):
        file_sample_rate, waveform = audio
        waveform = torch.tensor(
            waveform, dtype=torch.float
        )  # [samples, channels] or [samples] from Gradio Audio component
        waveform = self._preprocess_waveform(waveform)
        waveform = self._resample(
            waveform, file_sample_rate, feature_extractor.sampling_rate
        )
        audio, spec = self._make_spec_with_ast_extractor(
            feature_extractor, truncate, waveform
        )
        fig = SpectrogramGenerator.plot_spectrogram(
            feature_extractor.sampling_rate, spec, hop_length
        )
        return fig

    def _create_spec_data_from_filename(
        self, file_name, audio_class, feature_extractor, selection_method, truncate
    ):
        (
            file_name,
            audio_class,
            waveform,
            file_sample_rate,
        ) = self.dataset_handler.load_file(file_name, audio_class, selection_method)
        logger.info(waveform.shape)
        waveform = self._preprocess_waveform(waveform)
        waveform = self._resample(
            waveform, file_sample_rate, feature_extractor.sampling_rate
        )
        audio, spec = self._make_spec_with_ast_extractor(
            waveform, feature_extractor, truncate
        )
        return spec, audio, file_name, audio_class

    def _preprocess_waveform(self, waveform):
        logger.info(f"{waveform.shape=}")
        # convert [samples] to [samples, channels]
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(1)
        # transpose it to [channels, samples]
        waveform = waveform.transpose(0, 1)
        # convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _resample(self, audio, file_sample_rate, sample_rate):
        logger.info(f"{audio.shape=} {file_sample_rate=}")

        if file_sample_rate != sample_rate:
            resampler = Resample(file_sample_rate, sample_rate)
            audio = resampler(audio)
        else:
            logger.info("No resampling")
        logger.info(audio.shape)
        return audio

    def _make_spec_with_ast_extractor(
        self, waveform, feature_extractor, truncate=False, hop_length=160
    ):
        logger.info(waveform.shape)  # [1, samples]
        inputs = feature_extractor(
            waveform.numpy(),
            return_tensors="pt",
            padding="max_length",
        )
        spec = inputs["input_values"]  # [batch(1), frames, freq_bins]
        logger.info(spec.shape)
        spec = torch.squeeze(spec, 0)  # [frames, freq_bins]
        logger.info(spec.shape)
        if truncate:
            logger.info(f"{waveform.shape[1]=} {hop_length=}")
            actual_frames = torch.ceil(
                torch.tensor(waveform.shape[1] / hop_length)
            ).int()
            logger.info(actual_frames)
            spec = spec[:actual_frames, :]

        logger.info(f"{spec.shape=}")
        return waveform, spec

    def make_spec_from_audio_tuple(self, audio, feature_extractor):
        # input is (sr, waveform), so convert waveform to tensor
        waveform = torch.tensor(audio[1])
        waveform = torch.unsqueeze(waveform, 0)
        raw_audio, spec = self._make_spec_with_ast_extractor(
            waveform, feature_extractor, truncate=False
        )
        return feature_extractor.sampling_rate, raw_audio, spec
