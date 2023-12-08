import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio
from librosa.display import specshow
from spectrogram_generator import SpectrogramGenerator
from torchaudio.transforms import Resample
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from urban_sound_dataset_handler import UrbanSoundDatasetHandler

import gradio as gr

model_mapping = {
    "AST": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "AST2": "MIT/ast-finetuned-speech-commands-v2",
}
models = {
    name: AutoModelForAudioClassification.from_pretrained(
        model_mapping[name],
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    for name in model_mapping.keys()
}
feature_extractors = {
    name: AutoFeatureExtractor.from_pretrained(model_mapping[name], max_length=1024)
    for name in model_mapping.keys()
}
print(feature_extractors.keys())


def resample(audio, file_sr, input_sr):
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if file_sr != input_sr:
        resampler = Resample(file_sr, input_sr)
        audio = resampler(audio)

    num_samples = audio.shape[-1]
    total_duration = num_samples / input_sr

    return audio, num_samples, total_duration


def preprocess(waveform, file_sr, input_sr):
    audio, _, duration = resample(waveform, file_sr, input_sr)
    spec_generator = SpectrogramGenerator(input_sr)
    spec = spec_generator.generate_mel_spectrogram(audio, input_sr)
    spec = spec_generator.normalize_spectrogram(spec)
    return audio, spec


def preprocess_with_ast_feature_extractor(
    waveform, file_sr, output_sr, feature_extractor
):
    print("preprocessing")
    raw_audio, _, duration = resample(waveform, file_sr, output_sr)
    print(raw_audio.shape)
    inputs = feature_extractor(
        raw_audio.numpy(),
        sampling_rate=output_sr,
        padding="max_length",
        return_tensors="pt",
    )
    spec = inputs["input_values"]
    print(type(spec))
    spec = torch.squeeze(spec, 0)
    spec = torch.transpose(spec, 0, 1)
    actual_frames = np.ceil(len(raw_audio[0]) / 160).astype(int)
    spec = spec[:, :actual_frames]

    return raw_audio, spec


def plot_spectrogram(input_sr, spec, hop_length):
    plt.close()
    fig, ax = plt.subplots(figsize=(5, 2))
    _ = specshow(
        spec.numpy(),
        sr=input_sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    return fig


def get_feature_extractor(model_short_name):
    feature_extractor = feature_extractors[model_short_name]
    if model_short_name == "AST" or model_short_name == "AST2":
        extractor_frame_shift = 10
        extractor_hop_length = feature_extractor.sampling_rate * (
            extractor_frame_shift / 1000
        )
    return feature_extractor, extractor_hop_length


def process(
    file_name=None, audio_class=None, model_short_name="AST", selection_method="random"
):
    file_name, audio_class, waveform, file_sr = dataset_handler.load_file(
        file_name, audio_class, selection_method
    )
    if model_short_name in model_mapping:
        feature_extractor, hop_length = get_feature_extractor(model_short_name)
        input_sr = feature_extractor.sampling_rate
        audio, spec = preprocess_with_ast_feature_extractor(
            waveform, file_sr, input_sr, feature_extractor
        )
    else:
        input_sr = 22050
        audio, spec = preprocess(waveform, file_sr, input_sr)
        hop_length = 256
    fig = plot_spectrogram(input_sr, spec, hop_length)
    return fig, audio[0].numpy(), file_name, audio_class, input_sr


def predict(audio, model_short_name):
    print("predict")
    sr, waveform = audio
    waveform = torch.tensor(waveform, dtype=torch.float32)
    waveform = torch.unsqueeze(waveform, 0)
    print("supplied sr", sr)
    # model = models[model_mapping[model_short_name]]
    feature_extractor, _ = get_feature_extractor(model_short_name)
    model = models[model_short_name]
    input_sr = feature_extractor.sampling_rate
    _, spec = preprocess_with_ast_feature_extractor(
        waveform, input_sr, input_sr, feature_extractor
    )
    with torch.no_grad():
        logits = model(torch.unsqueeze(spec, 0)).logits
    return logits.shape


def generate_gradio_elements(
    file_name, class_picker, model_short_name, selection_method
):
    fig, audio, file_name, audio_class, input_sr = process(
        file_name, class_picker, model_short_name, selection_method
    )
    fig = gr.Plot(value=fig)
    audio = gr.Audio(value=(input_sr, audio))
    file_name = gr.Textbox(value=file_name)
    class_picker = gr.Dropdown(value=audio_class)
    return fig, audio, file_name, class_picker


def generate_gradio_elements_filename(file_name, class_picker, model_short_name, _):
    return generate_gradio_elements(
        file_name, class_picker, model_short_name, "filename"
    )


dataset_handler = UrbanSoundDatasetHandler()
classes = list(dataset_handler.class_to_class_id.keys())
choices = [
    ("by slice_file_name", "filename"),
    ("randomly from class", "class"),
    ("randomly from entire dataset", "random"),
]

with gr.Blocks() as demo:
    with gr.Row():
        selection_method = gr.Radio(label="pick file", choices=choices, value="random")
        model_short_name = gr.Dropdown(
            choices=["AST", "AST2", "local"], value="AST", label="Choose a model"
        )
    with gr.Row():
        file_name = gr.Textbox(label="slice_file_name in dataset")
        class_picker = gr.Dropdown(
            choices=classes, label="Choose a class", value=classes[-1]
        )
        gen_button = gr.Button("Get Spec")
    with gr.Row():
        spec = gr.Plot(container=True)
        my_audio = gr.Audio(interactive=True)

    gen_button.click(
        fn=generate_gradio_elements,
        inputs=[file_name, class_picker, model_short_name, selection_method],
        outputs=[spec, my_audio, file_name, class_picker],
    )
    # model_short_name.change(
    #     fn=generate_gradio_elements_filename,
    #     inputs=[file_name, class_picker, model_short_name, selection_method],
    #     outputs=[spec, my_audio, file_name, class_picker],
    # )
    gr.Examples(
        examples=[["100263-2-0-117.wav"], ["100852-0-0-0.wav"]],
        inputs=[file_name, class_picker, model_short_name, selection_method],
        outputs=[spec, my_audio, file_name, class_picker],
        run_on_click=True,
        fn=generate_gradio_elements,
    )
    infer = gr.Button("classify")
    infer_out = gr.TextArea("")
    infer.click(fn=predict, inputs=[my_audio, model_short_name], outputs=[infer_out])

if __name__ == "__main__":
    demo.launch()


def test_process():
    fig, audio, file_name, audio_class, input_sr = process(
        file_name="138031-2-0-45.wav", model_short_name="AST2"
    )
    plt.show()
    Audio(audio, rate=input_sr)
