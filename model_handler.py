import logging

import torch
from mel_band_filter import MelBandFilter
from spectrogram_generator import SpectrogramGenerator
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available else "cpu"


class ModelHandler:
    """
    A handler class for managing and utilizing audio classification models.

    This class is designed to handle various tasks related to audio classification models,
    including loading models, extracting features from audio data, classifying audio samples,
    and creating filtered spectrograms and audio segments.

    Attributes:
    ------------
    model_mapping : dict
        A dictionary mapping model short names to their respective model paths.
    models : dict
        A dictionary of loaded audio classification models.
    feature_extractors : dict
        A dictionary of feature extractors corresponding to the models.
    dataset_handler : DatasetHandler
        An instance of a dataset handler for managing dataset-specific operations.

    Methods:
    --------
    get_feature_extractor(model_short_name):
        Returns the feature extractor and hop length for the specified model.

    classify_audio_sample(spec, model_short_name):
        Classifies a spectrogram using the specified model and returns the logits, predicted class ID, and predicted class name.

    create_filtered_spec_and_audio(spec, model_short_name, actual_class_id, num_time_slices=10, audio=None, actual_length=400):
        Creates and returns filtered spectrogram variants, most valuable segment, padded spectrogram, and filtered audio for a given audio sample.
    """

    def __init__(self, dataset_handler):
        self.model_mapping = {
            "AST": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "AST2": "MIT/ast-finetuned-speech-commands-v2",
            "FT": "danavery/ast-finetune-urbansound8k",
        }
        self.models = {}
        self.feature_extractors = {}
        self.dataset_handler = dataset_handler

    def get_feature_extractor(self, model_short_name):
        if model_short_name not in self.feature_extractors:
            self.feature_extractors[
                model_short_name
            ] = AutoFeatureExtractor.from_pretrained(
                self.model_mapping[model_short_name]
            )
        feature_extractor = self.feature_extractors[model_short_name]
        if (
            model_short_name == "AST"
            or model_short_name == "AST2"
            or model_short_name == "FT"
        ):
            extractor_frame_shift = 10
            extractor_hop_length = feature_extractor.sampling_rate * (
                extractor_frame_shift / 1000
            )
        return feature_extractor, extractor_hop_length

    def classify_audio_sample(self, spec, model_short_name):
        model = self._get_model(model_short_name)
        logits, predicted = self._perform_classification(spec, model)
        predicted_class = self.dataset_handler.class_id_to_class[predicted.item()]
        return (logits, predicted_class)

    def create_filtered_spec_and_audio(
        self,
        spec,
        model_short_name,
        actual_class_id,
        num_time_slices=10,
        num_mel_slices=2,
        sample_rate=16000,
        audio=None,
        actual_length=400,
    ):
        mel_bands = spec.shape[1]
        time_slice_size = actual_length // num_time_slices
        spec_variants = self._generate_spec_variants(
            spec, num_time_slices, num_mel_slices, sample_rate, time_slice_size
        )
        logits = self._run_inference_on_variants(model_short_name, spec_variants)
        (
            most_valuable,
            most_valuable_time,
            most_valuable_mel_slice,
        ) = self._find_most_valuable_segment(actual_class_id, num_mel_slices, logits)
        logger.info(f"{most_valuable_mel_slice=}, {most_valuable_time=}")
        most_valuable_spec = SpectrogramGenerator.pad_spec(
            spec,
            time_slice_size,
            most_valuable_time,
            mel_bands,
            num_mel_portions=num_mel_slices,
            most_valuable_mel_index=most_valuable_mel_slice,
        )
        val_audio = ModelHandler.filter_audio(
            num_time_slices,
            audio,
            mel_bands,
            num_mel_slices,
            most_valuable_mel_slice,
            most_valuable_time,
        )
        return (
            most_valuable_spec,
            val_audio,
            num_time_slices,
            most_valuable_time,
            most_valuable_mel_slice,
        )

    def _find_most_valuable_segment(self, actual_class_id, num_mel_slices, logits):
        most_valuable = ModelHandler.get_most_valuable(actual_class_id, logits)
        logger.info(f"{most_valuable=}")
        most_valuable_time, most_valuable_mel_slice = divmod(
            most_valuable, num_mel_slices
        )
        return most_valuable, most_valuable_time, most_valuable_mel_slice

    def _generate_spec_variants(
        self, spec, num_time_slices, num_mel_slices, sample_rate, time_slice_size
    ):
        spec_variants = SpectrogramGenerator.split_spectrogram(
            spec, num_time_slices, time_slice_size, sample_rate, num_mel_slices
        )

        return spec_variants

    def get_model_device(model):
        # Get the device of the first parameter of the model
        # All parameters should be on the same device
        return next(model.parameters()).device

    def _run_inference_on_variants(self, model_short_name, spec_variants):
        model = self.models[model_short_name]
        logger.info(ModelHandler.get_model_device(model))
        model.eval()
        spec_variants = spec_variants.to(device)
        with torch.no_grad():
            logits = model(spec_variants).logits
        return logits

    def _perform_classification(self, spec, model):
        model.eval()
        batched = torch.unsqueeze(spec, 0).to(device)
        with torch.no_grad():
            logits = model(batched).logits
            _, predicted = torch.max(logits, 1)
        logger.info(self.dataset_handler.class_id_to_class)
        return logits, predicted

    def _get_model(self, model_short_name):
        if model_short_name not in self.models:
            self.models[
                model_short_name
            ] = AutoModelForAudioClassification.from_pretrained(
                self.model_mapping[model_short_name],
                num_labels=10,
                ignore_mismatched_sizes=True,
            ).to(
                device
            )
        return self.models[model_short_name]

    @staticmethod
    def get_most_valuable(actual_class_id, logits):
        logit_mins = torch.min(logits[:, actual_class_id], dim=0)
        most_valuable = logit_mins.indices.item()
        return most_valuable

    @staticmethod
    def filter_audio(
        num_time_slices,
        audio,
        mel_bands,
        num_mel_slices,
        most_valuable_mel_slice,
        most_valuable_time_index,
        sample_rate=16000,
    ):
        mel_range = ModelHandler.mel_slice_to_mel_range(
            mel_bands, num_mel_slices, most_valuable_mel_slice
        )
        mel_filter = MelBandFilter(mel_bands, sample_rate)
        val_audio = mel_filter.filter_time_slice(
            audio, mel_range, num_time_slices, most_valuable_time_index
        )
        return val_audio

    @staticmethod
    def mel_slice_to_mel_range(num_mels, num_mel_slices, mel_slice_index):
        mel_slice_size = num_mels // num_mel_slices
        mel_range_start = mel_slice_size * mel_slice_index
        mel_range_end = mel_range_start + mel_slice_size
        return (mel_range_start, mel_range_end)

    @staticmethod
    def calculate_softmax_probs(logits, class_names):
        sm = torch.nn.Softmax()
        probs = sm(logits[0]).cpu().numpy()
        class_probs = dict(zip(class_names, probs))
        return class_probs
