import logging
import torch
from spectrogram_generator import SpectrogramGenerator
from mel_band_filter import MelBandFilter

import gradio as gr

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GradioUIGenerator:
    def __init__(self, extractor, model_handler, dataset_handler):
        self.extractor = extractor
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.mel_filter = MelBandFilter(128, 16000)

    def update_gradio_elements(
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
        file_name_element = gr.Textbox(value=file_name)
        actual_file_name = gr.Text(value=file_name)
        actual_class = gr.Text(value=audio_class)
        class_picker_element = gr.Dropdown(value=audio_class)

        return (
            fig,
            audio,
            file_name_element,
            class_picker_element,
            actual_file_name,
            actual_class,
        )

    def update_gradio_elements_from_filename(
        self, file_name, class_picker, model_short_name, _
    ):
        return self.update_gradio_elements(
            file_name, class_picker, model_short_name, "filename"
        )

    def classify_audio_sample(
        self, waveform, model_short_name, file_name, class_picker
    ):
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        class_id = self.dataset_handler.class_to_class_id[class_picker]
        input_sr = feature_extractor.sampling_rate
        waveform = torch.tensor(waveform[1])
        waveform = torch.unsqueeze(waveform, 0)
        raw_audio, spec = self.extractor.make_spec_from_ast(
            waveform, input_sr, input_sr, feature_extractor, truncate=False
        )
        logits, predicted, predicted_class = self.model_handler.classify_audio_sample(
            spec, model_short_name
        )
        (
            _,
            most_valuable,
            most_valuable_spec,
            most_valuable_audio,
            prop,
        ) = self.model_handler.create_spec_variants(
            spec, model_short_name, class_id, 5, raw_audio[0]
        )
        fig = SpectrogramGenerator.plot_spectrogram(
            input_sr, spec[0:400, :].transpose(0, 1), 160
        )
        val_fig = SpectrogramGenerator.plot_spectrogram(
            input_sr, most_valuable_spec[0:400, :].transpose(0, 1), 160
        )
        return (
            f"{logits}\n{predicted}\n{predicted_class}\nmost valuable segment: {most_valuable}/{prop-1}",
            gr.Audio((input_sr, raw_audio[0].numpy())),
            gr.Plot(fig),
            gr.Plot(val_fig),
            gr.Audio((input_sr, most_valuable_audio.numpy())),
        )
