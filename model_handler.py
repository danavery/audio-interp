import logging

import torch
from mel_band_filter import MelBandFilter
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, dataset_handler):
        self.model_mapping = {
            "AST": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "AST2": "MIT/ast-finetuned-speech-commands-v2",
            "FT": "danavery/ast-finetune-urbansound8k",
        }
        self.models = {
            name: AutoModelForAudioClassification.from_pretrained(
                self.model_mapping[name],
                num_labels=10,
                ignore_mismatched_sizes=True,
            )
            for name in self.model_mapping.keys()
        }
        self.feature_extractors = {
            name: AutoFeatureExtractor.from_pretrained(self.model_mapping[name])
            for name in self.model_mapping.keys()
        }
        self.dataset_handler = dataset_handler

    def get_feature_extractor(self, model_short_name):
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
        model = self.models[model_short_name]
        model.eval()
        batched = torch.unsqueeze(spec, 0)
        with torch.no_grad():
            logits = model(batched).logits
            _, predicted = torch.max(logits, 1)
        logger.info(self.dataset_handler.class_id_to_class)
        predicted_class = self.dataset_handler.class_id_to_class[predicted.item()]
        return (logits, predicted.item(), predicted_class)

    def create_spec_variants(
        self, spec, model_short_name, actual_class_id, prop=10, audio=None
    ):
        logger.info(spec.shape)
        timesteps, mel_bands = spec.shape
        segment_size = 400 // prop
        spec_variants = []
        model = self.models[model_short_name]
        model.eval()

        for i in range(prop):
            start = i * segment_size
            end = start + segment_size
            modified_spec = spec.clone()
            modified_spec[start:end, :] = 0.4670
            spec_variants.append(modified_spec)
        spec_variants = torch.stack(spec_variants)
        with torch.no_grad():
            logits = model(spec_variants).logits

        logit_mins = torch.min(logits[:, actual_class_id], dim=0)
        most_valuable = logit_mins.indices.item()
        most_valuable_spec = spec.clone()
        start = most_valuable * segment_size
        end = start + segment_size
        # most_valuable_spec = spec[start:end, :]
        most_valuable_spec[0:start, :] = 0.0
        most_valuable_spec[end:, :] = 0.0

        mel_filter = MelBandFilter(128, 16000)
        val_audio = mel_filter.filter_time_slice(audio, (100, 127), prop, most_valuable)
        return spec_variants, most_valuable, most_valuable_spec, val_audio, prop
