# %%
#
# active development moved to app.py
#

from collections import defaultdict
import random

from librosa.display import specshow
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample
import gradio as gr
from transformers import AutoFeatureExtractor
import numpy as np
from IPython.display import Audio


# %%
feature_extractor_names = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "MIT/ast-finetuned-speech-commands-v2",
]
feature_extractors = {
    name: AutoFeatureExtractor.from_pretrained(name, max_length=1024)
    for name in feature_extractor_names
}

dataset = load_dataset("danavery/urbansound8k")
filename_to_index = defaultdict(int)
class_to_class_id = defaultdict(int)
class_id_files = defaultdict(list)
for index, item in enumerate(dataset["train"]):
    filename = item["slice_file_name"]
    class_name = item["class"]
    class_id = int(item["classID"])

    filename_to_index[filename] = index
    class_to_class_id[class_name] = class_id
    class_id_files[class_id].append(filename)


# %%
def fetch_random():
    example_index = random.randint(0, len(dataset["train"]) - 1)
    example = dataset["train"][example_index]
    return example


# %%
def get_random_index_by_class(audio_class):
    class_id = class_to_class_id[audio_class]
    filenames = class_id_files.get(class_id, [])
    selected_filename = random.choice(filenames)
    index = filename_to_index.get(selected_filename)
    return index


# %%
def fetch_example(file_name=None, audio_class=None):
    if file_name:
        example = dataset["train"][filename_to_index[file_name]]
    elif audio_class:
        example = dataset["train"][get_random_index_by_class(audio_class)]
    else:
        example = fetch_random()

    waveform = torch.tensor(example["audio"]["array"]).float()
    waveform = torch.unsqueeze(waveform, 0)
    sr = example["audio"]["sampling_rate"]
    slice_file_name = example["slice_file_name"]
    audio_class = example["class"]
    return waveform, sr, slice_file_name, audio_class


# %%
def make_mel_spectrogram(
    audio: torch.Tensor, sample_rate, hop_length=256, n_fft=512, n_mels=64
) -> torch.Tensor:
    spec_transformer = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = spec_transformer(audio).squeeze(0)
    amplitude_to_db_transformer = AmplitudeToDB()
    mel_spec_db = amplitude_to_db_transformer(mel_spec)
    return mel_spec_db


def normalize_spectrogram(spec):
    spectrogram = (spec - torch.min(spec)) / (torch.max(spec) - torch.min(spec))
    return spectrogram


def generate_spectrogram(audio, input_sr):
    spectrogram = make_mel_spectrogram(audio, input_sr)
    return spectrogram


# %%
def resample(audio, file_sr, input_sr):
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    resampler = Resample(file_sr, input_sr)
    audio = resampler(audio)

    num_samples = audio.shape[-1]
    total_duration = num_samples / input_sr

    return audio, num_samples, total_duration


# %%
def preprocess(waveform, file_sr, input_sr):
    audio, _, duration = resample(waveform, file_sr, input_sr)
    spec = generate_spectrogram(audio, input_sr)
    spec = normalize_spectrogram(spec)
    return audio, spec


# %%
def preprocess_with_ast_feature_extractor(
    waveform, file_sr, output_sr, feature_extractor
):
    raw_audio, _, duration = resample(waveform, file_sr, output_sr)
    inputs = feature_extractor(
        raw_audio.numpy(),
        sampling_rate=output_sr,
        padding="max_length",
        return_tensors="pt",
    )
    spec = inputs["input_values"]
    spec = torch.squeeze(spec, 0)
    spec = torch.transpose(spec, 0, 1)
    actual_frames = np.ceil(len(raw_audio[0]) / 160).astype(int)
    spec = spec[:, :actual_frames]

    return raw_audio, spec


# %%
def load_file(file_name, audio_class, selection_method):
    if selection_method == "filename":
        waveform, file_sr, file_name, audio_class = fetch_example(
            file_name=file_name, audio_class=None
        )
    elif selection_method == "class":
        waveform, file_sr, file_name, audio_class = fetch_example(
            file_name=None, audio_class=audio_class
        )
    else:
        waveform, file_sr, file_name, audio_class = fetch_example(
            file_name=None, audio_class=None
        )
    return file_name, audio_class, waveform, file_sr


# %%
def plot_spectrogram(input_sr, spec, hop_length):
    fig, ax = plt.subplots(figsize=(5, 2))
    img = specshow(
        spec.numpy(),
        sr=input_sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    return fig


# %%
def get_feature_extractor(feature_extractor_type):
    feature_extractor = feature_extractors[feature_extractor_type]
    if (
        feature_extractor_type == "MIT/ast-finetuned-audioset-10-10-0.4593"
        or feature_extractor_type == "MIT/ast-finetuned-speech-commands-v2"
    ):
        extractor_frame_shift = 10
        extractor_hop_length = feature_extractor.sampling_rate * (
            extractor_frame_shift / 1000
        )
    return feature_extractor, extractor_hop_length


# %%
def process(file_name=None, audio_class=None, model="AST", selection_method="randomly"):
    file_name, audio_class, waveform, file_sr = load_file(
        file_name, audio_class, selection_method
    )

    if model == "AST":
        feature_extractor, hop_length = get_feature_extractor(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        input_sr = feature_extractor.sampling_rate
        audio, spec = preprocess_with_ast_feature_extractor(
            waveform, file_sr, input_sr, feature_extractor
        )
    elif model == "AST2":
        feature_extractor, hop_length = get_feature_extractor(
            "MIT/ast-finetuned-speech-commands-v2"
        )
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


# %%
fig, audio, file_name, audio_class, input_sr = process(
    file_name="138031-2-0-45.wav", model="AST2"
)
plt.show()
Audio(audio, rate=input_sr)


# %%
def generate_gradio_elements(file_name, class_picker, model, selection_method):
    fig, audio, file_name, audio_class, input_sr = process(
        file_name, class_picker, model, selection_method
    )
    fig = gr.Plot(value=fig)
    audio = gr.Audio(value=(input_sr, audio))
    file_name = gr.Textbox(value=file_name)
    class_picker = gr.Dropdown(value=audio_class)
    return fig, audio, file_name, class_picker


def generate_gradio_elements_filename(file_name, class_picker, model, selection_method):
    return generate_gradio_elements(file_name, class_picker, model, "filename")


# %%
spec = process("137969-2-0-37.wav")

# %%
classes = list(class_to_class_id.keys())
choices = [
    ("by filename", "filename"),
    ("randomly from class", "class"),
    ("randomly from entire dataset", "random"),
]

with gr.Blocks() as demo:
    with gr.Row():
        selection_method = gr.Radio(label="pick file", choices=choices, value="random")
        model = gr.Dropdown(
            choices=["AST", "AST2", "local"], value="local", label="Choose a model"
        )
    with gr.Row():
        file_name = gr.Textbox(label="slice_file_name in dataset")
        class_picker = gr.Dropdown(
            choices=classes, label="Choose a category", value=classes[-1]
        )
        gen_button = gr.Button("Get Spec")
    with gr.Row():
        spec = gr.Plot(container=True)
        my_audio = gr.Audio()

    gen_button.click(
        fn=generate_gradio_elements,
        inputs=[file_name, class_picker, model, selection_method],
        outputs=[spec, my_audio, file_name, class_picker],
    )
    model.change(
        fn=generate_gradio_elements_filename,
        inputs=[file_name, class_picker, model, selection_method],
        outputs=[spec, my_audio, file_name, class_picker],
    )
    gr.Examples(
        examples=[["100263-2-0-117.wav"], ["100852-0-0-0.wav"]],
        inputs=[file_name, class_picker, model, selection_method],
        outputs=[spec, my_audio, file_name, class_picker],
        run_on_click=True,
        fn=generate_gradio_elements,
    )

demo.launch()
# %%
