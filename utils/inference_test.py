# %%
# %load_ext autoreload
# %autoreload 2
import csv
import logging

import torch
from audio_file_feature_extractor import AudioFileFeatureExtractor
# from IPython.display import Audio
from model_handler import ModelHandler
from transformers import ASTFeatureExtractor
from urban_sound_dataset_handler import UrbanSoundDatasetHandler

logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel("CRITICAL")
# %%

dataset_handler = UrbanSoundDatasetHandler()
model_handler = ModelHandler(dataset_handler)
feature_extractor = model_handler.feature_extractors["FT"]
# logger.info("pre-affe")
affe = AudioFileFeatureExtractor(model_handler, dataset_handler)
# logger.info("post-affe")
# %%

filenames = []
with open("/home/davery/UrbanSound8K.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["fold"] == "5":
            filenames.append(row["slice_file_name"])

# %%
correct = 0
start = 0
n = 100

for index, filename in enumerate(filenames[start : start + n]):
    slice_file_name, audio_class, waveform, sr = dataset_handler.load_file(
        file_name=filename, audio_class=None, selection_method="filename"
    )
    # Audio(waveform.numpy(), rate=sr)
    logger.info(waveform[:10])
    waveform = torch.unsqueeze(waveform, 0)
    _, spec = affe.make_spec_from_ast(waveform, sr, 16000, ASTFeatureExtractor(), False)
    logger.info(spec.shape)
    logger.info(spec[0])
    logits, predicted, predicted_class = model_handler.classify_audio_sample(spec, "FT")
    logger.info(slice_file_name)
    print(audio_class)
    print(predicted, predicted_class)
    if audio_class == predicted_class:
        print("CORRECT")
        correct += 1
    else:
        print("WRONG")
    print(correct, "/", index + 1)
# Audio(waveform.numpy(), rate=sr)
