import gradio as gr
from spectrogram_generator import SpectrogramGenerator
import torch


class GradioUIGenerator:
    def __init__(self, extractor, model_handler, dataset_handler):
        self.extractor = extractor
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler

    def create_gradio_elements(
        self, file_name, class_picker, model_short_name, selection_method
    ):
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        fig, audio, file_name, audio_class, input_sr = self.extractor.make_spec_plot(
            file_name, class_picker, feature_extractor, hop_length, selection_method
        )
        fig = gr.Plot(value=fig)
        audio = gr.Audio(value=(input_sr, audio))
        file_name = gr.Textbox(value=file_name)

        class_picker = gr.Dropdown(value=audio_class)
        return fig, audio, file_name, class_picker

    def create_gradio_elements_from_filename(
        self, file_name, class_picker, model_short_name, _
    ):
        return self.create_gradio_elements(
            file_name, class_picker, model_short_name, "filename"
        )

    def classify_audio_sample(self, waveform, model_short_name):
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        input_sr = feature_extractor.sampling_rate
        waveform = torch.tensor(waveform[1])
        waveform = torch.unsqueeze(waveform, 0)
        raw_audio, spec = self.extractor.make_spec_from_ast(
            waveform, input_sr, input_sr, feature_extractor, truncate=False
        )
        logits, predicted, predicted_class = self.model_handler.classify_audio_sample(
            spec, model_short_name
        )
        fig = SpectrogramGenerator.plot_spectrogram(input_sr, spec.transpose(0, 1), 160)
        return (
            f"{logits}\n{predicted}\n{predicted_class}",
            gr.Audio((input_sr, raw_audio[0].numpy())),
            gr.Plot(fig),
        )
