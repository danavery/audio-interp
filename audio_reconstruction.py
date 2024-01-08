import logging

import torch
import torchaudio
import torchaudio.transforms as T
from audio_file_feature_extractor import AudioFileFeatureExtractor
from gradio_ui_generator import GradioUIGenerator
from model_handler import ModelHandler
from urban_sound_dataset_handler import UrbanSoundDatasetHandler

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

dataset_handler = UrbanSoundDatasetHandler(regenerate=False)
model_handler = ModelHandler(dataset_handler)
extractor = AudioFileFeatureExtractor(model_handler, dataset_handler)
gradio_ui = GradioUIGenerator(extractor, model_handler, dataset_handler)

slice_file_name, audio_class, waveform, file_sr = dataset_handler.load_file(file_name="31325-3-1-0.wav", selection_method="filename")
feature_extractor, hop_length = model_handler.get_feature_extractor(
    "FT",
)
raw_audio, spec = extractor.make_spec_with_ast_extractor(
    waveform, file_sr,  file_sr, feature_extractor, truncate=False
)
logger.info(spec.shape)
logger.info(spec)


def inverse_normalize_spectrogram(spectrogram):
    return spectrogram * (4.5689974 * 2) - 4.2677393


def audio_from_spec(log_mel_spec):
    logger.info("generating audio")
    # mel_spec = torch.exp(log_mel_spec)
    mel_spec = inverse_normalize_spectrogram(log_mel_spec)
    logger.info(f"{mel_spec.shape=}")
    # mel_spec = mel_spec[:400, :]
    mel_spec = torch.permute(mel_spec, (1, 0))
    logger.info(f"{mel_spec.shape=}")

    mel_to_linear = T.InverseMelScale(n_stft=201, n_mels=128, driver="gelsd")
    linear_spec = mel_to_linear(mel_spec)
    logger.info(f"{linear_spec.shape=}")
    logger.info(f"{torch.sum(linear_spec)=}")
    griffin_lim = T.GriffinLim(n_fft=400, n_iter=128, hop_length=160)
    # linear_spec = torch.permute(linear_spec, (1, 0))
    logger.info(f"{linear_spec.shape=}")
    waveform = griffin_lim(linear_spec)

    logger.info("generated audio")
    return waveform


new_waveform = audio_from_spec(spec)
logger.info(torch.sum(spec))
new_waveform = torch.unsqueeze(new_waveform, 0)
logger.info(new_waveform.shape)
logger.info(torch.sum(new_waveform))
logger.info(new_waveform)
torchaudio.save("output.wav", new_waveform, 16000)
