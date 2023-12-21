import logging

from audio_file_feature_extractor import AudioFileFeatureExtractor
from gradio_ui_generator import GradioUIGenerator
from model_handler import ModelHandler
from urban_sound_dataset_handler import UrbanSoundDatasetHandler

import gradio as gr

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def generate_demo(gradio_ui):
    classes = gradio_ui.classes
    class_ids = gradio_ui.class_ids
    logger.info(classes)
    logger.info(class_ids)
    choices = [
        ("by slice_file_name", "filename"),
        ("randomly from class", "class"),
        ("randomly from entire dataset", "random"),
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                model_short_name = gr.Dropdown(
                    choices=gradio_ui.model_handler.model_mapping,
                    value="FT",
                    label="Choose a model to generate spectrogram and run classification with:",
                    visible=False,
                )
                gr.HTML("<h2>1) Choose a file from the UrbanSound8K dataset:</h2>")
                selection_method = gr.Radio(
                    label="Sound clip lookup:", choices=choices, value="class"
                )

                gr.HTML("Fill this in and choose 'randomly from class' above")
                class_picker = gr.Dropdown(
                    choices=classes,
                    label="Choose a class",
                    value="dog_bark",
                )
                gr.HTML("OR<br><br>Fill this in and choose 'by slice_file_name'")
                file_name = gr.Textbox(label="slice_file_name in dataset")
                gen_button = gr.Button("Get Audio and Generate Spectrogram")
            with gr.Column(scale=2):
                spec = gr.Plot(container=True, label="Mel spectrogram of file")
                with gr.Column():
                    # file_name_actual = gr.Textbox(label="current file name")
                    # class_actual = gr.Textbox(label="current class")
                    my_audio = gr.Audio(interactive=True, label="File audio")
                gr.Examples(
                    fn=gradio_ui.update_gradio_elements,
                    label="Preselected examples:",
                    examples=[["100263-2-0-117.wav"], ["100852-0-0-0.wav"]],
                    inputs=[file_name, class_picker, model_short_name, selection_method],
                    outputs=[
                        spec,
                        my_audio,
                        file_name,
                        class_picker,
                        # file_name_actual,
                        # class_actual,
                    ],
                    run_on_click=True,
                )
        gen_button.click(
            fn=gradio_ui.update_gradio_elements,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[
                spec,
                my_audio,
                file_name,
                class_picker,
                # file_name_actual,
                # class_actual,
            ],
        )
        model_short_name.change(
            fn=gradio_ui.update_gradio_elements_from_filename,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[
                spec,
                my_audio,
                file_name,
                class_picker,
                # file_name_actual,
                # class_actual,
            ],
        )

        gr.HTML("<hr>")
        with gr.Row():
            with gr.Column(scale=0):
                gr.HTML("<h2>2) Then run the audio through the model:</h2>")
                num_time_slices = gr.Dropdown(
                    choices=range(1, 6), value=3, scale=1, label="Number of Time Slices"
                )
                num_mel_slices = gr.Dropdown(
                    choices=range(1, 6), value=3, scale=1, label="Number of Mel Slices"
                )
                infer = gr.Button("Classify full audio and all sub-slices", scale=1)
                infer_out = gr.TextArea(
                    lines=1,
                    value="",
                    label="Output",
                    scale=3,
                    interactive=False,
                    visible=False,
                )
                prediction = gr.TextArea(
                    lines=1, value="", interactive=False, label="Prediction (full audio)"
                )
            with gr.Column(scale=3):
                predictions = gr.Label(label="Class probabilities (full audio)", scale=3)
            # with gr.Column():
            #     infer_audio = gr.Audio(visible=False)
            #     infer_spec = gr.Plot(visible=False, container=True)
            with gr.Column(scale=2):
                gr.HTML("<h2>Removing this part of the spectrogram has the most impact on the prediction:</h2>")
                infer_most_val_spec = gr.Plot(
                    container=True,
                    label="Most valuable portion of spec for current prediction",
                )
                gr.HTML("<h2>Compare this audio to the full audio above:</h2>")
                infer_most_val_audio = gr.Audio(
                    label="Most valuable audio by time and frequency"
                )
        infer.click(
            fn=gradio_ui.classify,
            inputs=[
                my_audio,
                model_short_name,
                class_picker,
                num_time_slices,
                num_mel_slices,
            ],
            outputs=[
                infer_out,
                # infer_audio,
                # infer_spec,
                infer_most_val_spec,
                infer_most_val_audio,
                prediction,
                predictions,
            ],
        )
        return demo


def create_demo():
    dataset_handler = UrbanSoundDatasetHandler(regenerate=False)
    model_handler = ModelHandler(dataset_handler)
    extractor = AudioFileFeatureExtractor(model_handler, dataset_handler)
    gradio_ui = GradioUIGenerator(extractor, model_handler, dataset_handler)
    # demo = gradio_ui.generate_demo()
    demo = generate_demo(gradio_ui)
    return demo


demo = create_demo()
demo.launch()
