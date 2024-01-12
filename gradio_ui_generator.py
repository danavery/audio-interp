import logging

from mel_band_filter import MelBandFilter
from model_handler import ModelHandler
from spectrogram_generator import SpectrogramGenerator

import gradio as gr

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.DEBUG,
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

    def generate_demo(self):
        classes = self.classes
        class_ids = self.class_ids
        logger.info(classes)
        logger.info(class_ids)
        choices = [
            ("by filename", "filename"),
            ("randomly from class", "class"),
            ("randomly from entire dataset", "random"),
        ]

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    model_short_name = self.model_selector()  # not visible for now
                    selection_method = self.file_selector(choices)
                    class_picker = self.class_selector(classes)
                    file_name = self.filename_input()
                    gen_button = gr.Button("Get Audio and Generate Spectrogram")
                with gr.Column(scale=2):
                    spec = self.spectrogram_plot()
                    with gr.Column():
                        my_audio = self.audio_player()
                    self.example_selector(
                        model_short_name,
                        selection_method,
                        class_picker,
                        file_name,
                        spec,
                        my_audio,
                    )
            self.make_spec_button_handler(
                model_short_name,
                selection_method,
                class_picker,
                file_name,
                gen_button,
                spec,
                my_audio,
            )
            self.audio_input_handler(model_short_name, spec, my_audio)
            self.model_change_handler(
                model_short_name,
                selection_method,
                class_picker,
                file_name,
                spec,
                my_audio,
            )

            gr.HTML("<hr>")
            with gr.Row():
                with gr.Column(scale=0):
                    gr.HTML("<h2>2) Then run the audio through the model:</h2>")
                    num_time_slices, num_mel_slices = self.time_and_mel_slice_selector()
                    infer = self.classify_button()
                    infer_out = self.infer_out_unused()
                    prediction = self.prediction_output()
                with gr.Column(scale=3):
                    predictions = self.label_probabilities()
                with gr.Column(scale=3):
                    infer_most_val_spec = self.most_valuable_spec()
                    infer_most_val_audio = self.most_valuable_audio()
            gr.HTML("<hr>")
            gr.Markdown(self.get_intro_markdown())
            self.classify_click_handler(
                model_short_name,
                class_picker,
                my_audio,
                num_time_slices,
                num_mel_slices,
                infer,
                infer_out,
                prediction,
                predictions,
                infer_most_val_spec,
                infer_most_val_audio,
            )
            return demo

    def classify_click_handler(
        self,
        model_short_name,
        class_picker,
        my_audio,
        num_time_slices,
        num_mel_slices,
        infer,
        infer_out,
        prediction,
        predictions,
        infer_most_val_spec,
        infer_most_val_audio,
    ):
        infer.click(
            fn=self.classify,
            inputs=[
                my_audio,
                model_short_name,
                class_picker,
                num_time_slices,
                num_mel_slices,
            ],
            outputs=[
                infer_out,
                infer_most_val_spec,
                infer_most_val_audio,
                prediction,
                predictions,
            ],
        )

    def most_valuable_audio(self):
        gr.HTML("<h2>Compare this audio to the full audio above:</h2>")
        infer_most_val_audio = gr.Audio(
            label="Most valuable audio by time and frequency"
        )

        return infer_most_val_audio

    def most_valuable_spec(self):
        gr.HTML(
            "<h2>Removing this part of the spectrogram has the most impact on the prediction:</h2>"
        )
        infer_most_val_spec = gr.Plot(
            container=True,
            label="Most valuable portion of spec for current prediction",
        )

        return infer_most_val_spec

    def label_probabilities(self):
        predictions = gr.Label(label="Class probabilities (full audio)", scale=3)

        return predictions

    def prediction_output(self):
        prediction = gr.TextArea(
            lines=1,
            value="",
            interactive=False,
            label="Prediction (full audio)",
        )

        return prediction

    def infer_out_unused(self):
        infer_out = gr.TextArea(
            lines=1,
            value="",
            label="Output",
            scale=3,
            interactive=False,
            visible=False,
        )

        return infer_out

    def classify_button(self):
        infer = gr.Button("Classify full audio and all sub-slices", scale=1)
        return infer

    def time_and_mel_slice_selector(self):
        num_time_slices = gr.Dropdown(
            choices=range(1, 7),
            value=3,
            scale=1,
            label="Number of Time Slices",
        )
        num_mel_slices = gr.Dropdown(
            choices=range(1, 7),
            value=3,
            scale=1,
            label="Number of Mel Slices",
        )

        return num_time_slices, num_mel_slices

    def spectrogram_plot(self):
        spec = gr.Plot(container=True, label="Mel spectrogram of file")
        return spec

    def audio_player(self):
        my_audio = gr.Audio(interactive=True, label="File audio")
        return my_audio

    def model_change_handler(
        self,
        model_short_name,
        selection_method,
        class_picker,
        file_name,
        spec,
        my_audio,
    ):
        model_short_name.change(
            fn=self.update_gradio_elements_from_filename,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[
                spec,
                my_audio,
                file_name,
                class_picker,
            ],
        )

    def audio_input_handler(self, model_short_name, spec, my_audio):
        my_audio.stop_recording(
            fn=self.generate_spec_from_input,
            inputs=[model_short_name, my_audio],
            outputs=[spec],
        )
        my_audio.upload(
            fn=self.generate_spec_from_input,
            inputs=[model_short_name, my_audio],
            outputs=[spec],
        )

    def make_spec_button_handler(
        self,
        model_short_name,
        selection_method,
        class_picker,
        file_name,
        gen_button,
        spec,
        my_audio,
    ):
        gen_button.click(
            fn=self.display_spec_from_dataset,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[
                spec,
                my_audio,
                file_name,
                class_picker,
            ],
        )

    def example_selector(
        self,
        model_short_name,
        selection_method,
        class_picker,
        file_name,
        spec,
        my_audio,
    ):
        gr.Examples(
            fn=self.display_spec_from_dataset,
            label="Preselected examples:",
            examples=[["100263-2-0-117.wav"], ["100852-0-0-0.wav"]],
            inputs=[
                file_name,
                class_picker,
                model_short_name,
                selection_method,
            ],
            outputs=[
                spec,
                my_audio,
                file_name,
                class_picker,
            ],
            run_on_click=True,
        )

    def filename_input(self):
        gr.HTML("OR<br><br>Fill this in and choose 'by filename'")
        file_name = gr.Textbox(label="filename")
        return file_name

    def class_selector(self, classes):
        gr.HTML("Fill this in and choose 'randomly from class' above")
        class_picker = gr.Dropdown(
            choices=classes,
            label="Choose a class",
            value="dog_bark",
        )

        return class_picker

    def file_selector(self, choices):
        gr.HTML("<h2>1) Choose a file from the UrbanSound8K dataset:</h2>")
        selection_method = gr.Radio(
            label="Sound clip lookup:", choices=choices, value="class"
        )
        return selection_method

    def model_selector(self):
        model_short_name = gr.Dropdown(
            choices=self.model_handler.model_mapping,
            value="FT",
            label="Choose a model to generate spectrogram and run classification with:",
            visible=False,
        )
        return model_short_name

    def display_spec_from_dataset(
        self, file_name, class_picker, model_short_name, selection_method
    ):
        logger.info("dsfd")
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        (
            fig,
            audio,
            file_name,
            audio_class,
            input_sr,
        ) = self.extractor.make_spec_from_dataset(
            file_name, class_picker, feature_extractor, hop_length, selection_method
        )
        fig_element = gr.Plot(
            value=fig, label=f"Mel Spectrogram for {file_name} ({audio_class})"
        )
        audio_element = gr.Audio(value=(input_sr, audio[0]))
        logger.info(audio)
        file_name_element = gr.Textbox(value=file_name)
        actual_file_name = gr.Text(value=file_name)
        actual_class = gr.Text(value=audio_class)
        class_picker_element = gr.Dropdown(value=audio_class)

        return (
            fig_element,
            audio_element,
            file_name_element,
            class_picker_element,
            actual_file_name,
            actual_class,
        )

    def generate_spec_from_input(self, model_short_name, waveform):
        logger.info("gsfi")
        feature_extractor, hop_length = self.model_handler.get_feature_extractor(
            model_short_name
        )
        fig = self.extractor.make_spec_from_input(
            feature_extractor, hop_length, True, waveform
        )
        fig_element = gr.Plot(value=fig, label="Mel Spectrogram for your input")
        return fig_element

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
        ) = self.audio_file_feature_extractor.make_spec_from_audio_tuple(
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
        logger.info(f"{len(raw_audio[0])}")
        actual_length = int(len(raw_audio[0]) // hop_length)
        logger.info(f"{actual_length=}")
        most_valuable_spectrogram_segment = SpectrogramGenerator.plot_spectrogram(
            input_sr, most_valuable_spec[0:actual_length, :], hop_length
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

    def get_intro_markdown(self):
        with open("./README.md", "r") as file:
            intro_markdown = file.read()
            position = intro_markdown.find("# Playing")
            if position != -1:
                intro_markdown = intro_markdown[position:]
        return intro_markdown
