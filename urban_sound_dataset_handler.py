import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Audio, load_dataset

logger = logging.getLogger(__name__)
logger.propagate = True


class UrbanSoundDatasetHandler:
    def __init__(self, dataset_name="danavery/urbansound8k", regenerate=False):
        self.dataset = load_dataset(dataset_name)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.index_path = Path("us_indexes.pkl")
        self.regenerate = regenerate
        self.indexes = self._create_or_load_indexes()

    def load_file(self, file_name=None, audio_class=None, selection_method="random"):
        if selection_method == "filename":
            waveform, file_sr, slice_file_name, audio_class = self._get_audio_sample(
                file_name=file_name
            )
        elif selection_method == "class":
            waveform, file_sr, slice_file_name, audio_class = self._get_audio_sample(
                audio_class=audio_class
            )
        else:
            waveform, file_sr, slice_file_name, audio_class = self._get_audio_sample()
        return slice_file_name, audio_class, waveform, file_sr

    def _create_or_load_indexes(self):
        if self.index_path.is_file() and not self.regenerate:
            return self._load_indexes()
        else:
            return self._create_indexes()

    def _create_indexes(self):
        indexes = {
            "filename_to_index": {},
            "class_to_class_id": {},
            "class_id_to_class": {},
            "class_id_files": defaultdict(list),
            "file_to_class_id": {},
        }

        for index, item in enumerate(self.dataset["train"]):
            slice_file_name = item["slice_file_name"]
            class_name = item["class"]
            class_id = item["classID"]

            indexes["filename_to_index"][slice_file_name] = index
            indexes["class_to_class_id"][class_name] = int(class_id)
            indexes["class_id_to_class"][int(class_id)] = class_name
            indexes["class_id_files"][int(class_id)].append(slice_file_name)
            indexes["file_to_class_id"][slice_file_name] = class_id

        with self.index_path.open("wb") as file:
            pickle.dump(indexes, file)
        return indexes

    def _load_indexes(self):
        with self.index_path.open("rb") as file:
            indexes = pickle.load(file)
            return indexes

    def _fetch_random_audio_example(self):
        example_index = random.randint(0, len(self.dataset["train"]) - 1)
        example = self.dataset["train"][example_index]
        return example

    def _fetch_random_example_by_class(self, audio_class):
        class_id = self.indexes["class_to_class_id"][audio_class]
        filenames = self.indexes["class_id_files"][class_id]
        selected_filename = random.choice(filenames)
        index = self.indexes["filename_to_index"][selected_filename]
        example = self.dataset["train"][index]
        return example

    def _fetch_example_by_filename(self, file_name):
        example = self.dataset["train"][self.indexes["filename_to_index"][file_name]]
        return example

    def _get_audio_sample(self, file_name=None, audio_class=None):
        if file_name:
            example = self._fetch_example_by_filename(file_name)
        elif audio_class:
            example = self._fetch_random_example_by_class(audio_class)
        else:
            example = self._fetch_random_audio_example()
        waveform = torch.tensor(example["audio"]["array"])
        sr = example["audio"]["sampling_rate"]
        slice_file_name = example["slice_file_name"]
        audio_class = example["class"]
        return waveform, sr, slice_file_name, audio_class


if __name__ == "__main__":
    UrbanSoundDatasetHandler(regenerate=True)
