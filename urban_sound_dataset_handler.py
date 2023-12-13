import os
import pickle
import random
from collections import defaultdict
import logging
import torch
from datasets import Audio, load_dataset

logger = logging.getLogger(__name__)
logger.propagate = True


class UrbanSoundDatasetHandler:
    def __init__(self, dataset_name="danavery/urbansound8k", regenerate=False):
        self.dataset = load_dataset(dataset_name)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.filename_to_index = defaultdict(int)
        self.class_to_class_id = {}
        self.class_id_to_class = {}
        self.class_id_files = defaultdict(list)
        self.file_name = "us_indexes.pkl"
        self.regenerate = regenerate
        self._index_dataset()

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

    def _index_dataset(self):
        # will need to update this serialization for HF Spaces use
        if os.path.isfile(self.file_name) and not self.regenerate:
            self._read_saved_indexes()
        else:
            self._create_saved_indexes()

    def _create_saved_indexes(self):
        for index, item in enumerate(self.dataset["train"]):
            self.filename_to_index[item["slice_file_name"]] = index
            self.class_to_class_id[item["class"]] = int(item["classID"])
            self.class_id_to_class[int(item["classID"])] = item["class"]
            self.class_id_files[int(item["classID"])].append(item["slice_file_name"])
        with open(self.file_name, "wb") as file:
            pickle.dump(
                (
                    self.filename_to_index,
                    self.class_to_class_id,
                    self.class_id_to_class,
                    self.class_id_files,
                ),
                file,
            )

    def _read_saved_indexes(self):
        with open(self.file_name, "rb") as file:
            (
                self.filename_to_index,
                self.class_to_class_id,
                self.class_id_to_class,
                self.class_id_files,
            ) = pickle.load(file)

    def _fetch_random_audio_example(self):
        example_index = random.randint(0, len(self.dataset["train"]) - 1)
        example = self.dataset["train"][example_index]
        return example

    def _fetch_random_example_by_class(self, audio_class):
        class_id = self.class_to_class_id[audio_class]
        filenames = self.class_id_files.get(class_id, [])
        selected_filename = random.choice(filenames)
        index = self.filename_to_index.get(selected_filename)
        example = self.dataset["train"][index]
        return example

    def _fetch_example_by_filename(self, file_name):
        example = self.dataset["train"][self.filename_to_index[file_name]]
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
