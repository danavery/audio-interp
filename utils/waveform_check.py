import logging
import warnings

import torch
from datasets import Audio, load_dataset
from transformers import ASTFeatureExtractor, AutoFeatureExtractor

# from torchaudio.transforms import Resample
# import librosa

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
torch.set_printoptions(threshold=10_000)
warnings.filterwarnings(
    "ignore",
    message="promote has been superseded by mode='default'.",
)

ft_feature_extractor = ASTFeatureExtractor()
in_feature_extractor = AutoFeatureExtractor.from_pretrained(
    "danavery/ast-finetune-urbansound8k"
)


def preprocess_function(examples):
    print("WHAT")
    audio_arrays = [x["array"] for x in examples["audio"]]
    logger.info(torch.tensor(audio_arrays[0][:10]))

    inputs = ft_feature_extractor(
        audio_arrays,
        sampling_rate=ft_feature_extractor.sampling_rate,
        padding="max_length",
        return_tensors="pt",
    )
    logger.info(inputs["input_values"][0][0])
    return


def finetune():
    dataset = load_dataset("danavery/urbansound8k")["train"]
    train_subset = dataset.filter(
        lambda example: example["slice_file_name"] == "94868-1-0-0.wav"
    )
    train_subset = train_subset.cast_column("audio", Audio(sampling_rate=16000))
    train_subset = train_subset.map(
        preprocess_function,
        batch_size=64,
        batched=True,
        num_proc=8,
        keep_in_memory=True,
    )


def inference():
    dataset = load_dataset("danavery/urbansound8k")
    dataset["train"] = dataset["train"].cast_column("audio", Audio(sampling_rate=16000))
    for example in dataset["train"]:
        if example["slice_file_name"] != "94868-1-0-0.wav":
            continue
        waveform = torch.tensor(example["audio"]["array"]).float()

        sr = example["audio"]["sampling_rate"]
        logger.info(sr)
        # waveform = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000)
        # resampler = Resample(sr, 16000)
        # waveform = resampler(waveform)
        logger.info(waveform[0:10])
        inputs = in_feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
        )
        spec = inputs["input_values"]
        logger.info(spec.shape)
        logger.info(spec[0][0])
        break


finetune()
inference()
