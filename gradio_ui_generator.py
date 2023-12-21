import logging

from audio_file_feature_extractor import AudioFileFeatureExtractor
from mel_band_filter import MelBandFilter
from model_handler import ModelHandler
from spectrogram_generator import SpectrogramGenerator

import gradio as gr

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GradioUIGenerator:
    def __init__(
        self, extractor, model_handler, dataset_handler, audio_file_feature_extractor
    ):
        self.extractor = extractor
        self.model_handler = model_handler
        self.dataset_handler = dataset_handler
        self.audio_file_feature_extractor = audio_file_feature_extractor
        self.mel_filter = MelBandFilter(128, 16000)
        self.classes = dataset_handler.class_to_class_id
        self.class_ids = dataset_handler.class_id_to_class
        self.classes_in_class_id_order = sorted(
            self.classes.keys(), key=lambda x: self.classes[x]
        )

    def update_gradio_elements(
        self, file_name, class_picker, model_short_name, selection_method
    ):
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        fig, audio, file_name, audio_class, input_sr = self.extractor.make_spec_plot(
            file_name, class_picker, feature_extractor, hop_length, selection_method
        )
        fig = gr.Plot(
            value=fig, label=f"Mel Spectrogram for {file_name} ({audio_class})"
        )
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

    def classify(
        self,
        waveform,
        model_short_name,
        class_picker,
        num_time_slices=8,
        num_mel_slices=3,
    ):
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )

        (
            input_sr,
            raw_audio,
            spec,
        ) = AudioFileFeatureExtractor.make_spec_from_waveform(
            waveform, feature_extractor
        )

        logits, predicted_class = self.model_handler.classify_audio_sample(
            spec, model_short_name
        )

        class_id = self.dataset_handler.class_to_class_id[class_picker]
        (
            most_valuable_spec,
            most_valuable_audio,
            num_time_slices,
            most_valuable_time,
            most_valuable_mel_slice,
        ) = self.model_handler.create_filtered_spec_and_audio(
            spec,
            model_short_name,
            class_id,
            num_time_slices,
            num_mel_slices,
            audio=raw_audio[0],
            sample_rate=input_sr,
        )

        most_valuable_spectrogram_segment = SpectrogramGenerator.plot_spectrogram(
            input_sr, most_valuable_spec[0:400, :].transpose(0, 1), hop_length
        )
        class_probs = ModelHandler.calculate_softmax_probs(
            logits, self.classes_in_class_id_order
        )
        return (
            f"{logits}\nmost valuable segment: time {most_valuable_time}/{num_time_slices-1}, mel {most_valuable_mel_slice}/{num_mel_slices-1}",
            gr.Plot(most_valuable_spectrogram_segment),
            gr.Audio((input_sr, most_valuable_audio.numpy())),
            gr.Textbox(predicted_class),
            gr.DataFrame(value=class_probs),
        )
