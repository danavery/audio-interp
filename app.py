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
    classes = gradio_ui.dataset_handler.class_to_class_id
    class_ids = gradio_ui.dataset_handler.class_id_to_class
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
                choices=gradio_ui.model_handler.model_mapping,
                value="FT",
                label="Choose a model",
            )
        with gr.Row():
            file_name = gr.Textbox(label="slice_file_name in dataset")
            class_picker = gr.Dropdown(
                choices=classes,
                label="Choose a class",
                value="dog_bark",
            )
            gen_button = gr.Button("Get Spec")
        with gr.Row():
            spec = gr.Plot(container=True)
            my_audio = gr.Audio(interactive=True)

        gen_button.click(
            fn=gradio_ui.create_gradio_elements,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[spec, my_audio, file_name, class_picker],
        )
        model_short_name.change(
            fn=gradio_ui.create_gradio_elements_from_filename,
            inputs=[file_name, class_picker, model_short_name, selection_method],
            outputs=[spec, my_audio, file_name, class_picker],
        )
        gr.Examples(
            fn=gradio_ui.create_gradio_elements,
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
            fn=gradio_ui.classify_audio_sample,
            inputs=[my_audio, model_short_name],
            outputs=[infer_out, infer_audio, infer_spec],
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
