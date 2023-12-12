import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


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
        with torch.no_grad():
            logits = model(torch.unsqueeze(spec, 0)).logits
            _, predicted = torch.max(logits, 1)
        print(logits)
        print(self.dataset_handler.class_id_to_class)
        predicted_class = self.dataset_handler.class_id_to_class[predicted.item()]
        return (logits, predicted, predicted_class)
