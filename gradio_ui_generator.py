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
        print(audio_class)
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
        fig = SpectrogramGenerator.plot_spectrogram(input_sr, spec, 160)
        return (
            f"{logits}\n{predicted.item()}\n{predicted_class}",
            gr.Audio((input_sr, raw_audio[0].numpy())),
            gr.Plot(fig),
        )

    def generate_demo(self):
        classes = self.dataset_handler.class_to_class_id
        class_ids = self.dataset_handler.class_id_to_class
        print(classes)
        print(class_ids)
        choices = [
            ("by slice_file_name", "filename"),
            ("randomly from class", "class"),
            ("randomly from entire dataset", "random"),
        ]

        with gr.Blocks() as demo:
            with gr.Row():
                selection_method = gr.Radio(
                    label="pick file", choices=choices, value="class"
                )
                model_short_name = gr.Dropdown(
                    choices=self.model_handler.model_mapping,
                    value="AST",
                    label="Choose a model",
                )
            with gr.Row():
                file_name = gr.Textbox(label="slice_file_name in dataset")
                class_picker = gr.Dropdown(
                    choices=classes,
                    label="Choose a class",
                    value=0,
                )
                gen_button = gr.Button("Get Spec")
            with gr.Row():
                spec = gr.Plot(container=True)
                my_audio = gr.Audio(interactive=True)

            gen_button.click(
                fn=self.create_gradio_elements,
                inputs=[file_name, class_picker, model_short_name, selection_method],
                outputs=[spec, my_audio, file_name, class_picker],
            )
            model_short_name.change(
                fn=self.create_gradio_elements_from_filename,
                inputs=[file_name, class_picker, model_short_name, selection_method],
                outputs=[spec, my_audio, file_name, class_picker],
            )
            gr.Examples(
                fn=self.create_gradio_elements,
                examples=[["100263-2-0-117.wav"], ["100852-0-0-0.wav"]],
                inputs=[file_name, class_picker, model_short_name, selection_method],
                outputs=[spec, my_audio, file_name, class_picker],
                run_on_click=True,
            )
            infer = gr.Button("classify")
            infer_out = gr.TextArea("")
            infer_audio = gr.Audio()
            infer_spec = gr.Plot(container=True)
            infer.click(
                fn=self.classify_audio_sample,
                inputs=[my_audio, model_short_name],
                outputs=[infer_out, infer_audio, infer_spec],
            )
        return demo
