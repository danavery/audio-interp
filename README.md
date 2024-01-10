---
title: Audio Model Interpretation
emoji: 👀
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.11.0
app_file: app.py
pinned: false
datasets:
    - danavery/urbansound8k

---
# Playing with AST Audio Classification Interpretation

## What is this?

It's a Gradio UI for a transformer model that tries to find the "most important" time and frequency slice of a given sound to its classification.
It uses the Urbansound8k [[original dataset]](https://urbansounddataset.weebly.com/urbansound8k.html) [[Hugging Face dataset]](https://huggingface.co/datasets/danavery/urbansound8K) [[paper]](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)
 dataset and a fine-tuned-by-me-for-Urbansound8k version of the pre-trained [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) Audio Spectrogram Transformer model from the authors of the original [AST paper](https://arxiv.org/abs/2104.01778).

## What do I do with it?

- Pick a sound from the Urbansound8k dataset.
    - For a completely random sound, select the option "randomly from the entire dataset".
    - If you just want an example of a particular class of sound, select the option "randomly from class" and choose a class from the dropdown.
    - If for some reason you know the name of a particular file, you can choose "slice_file_name" from the radio buttons and enter the name into the labelled textbox.
    - In all cases, then click on "Get Audio and Generate Spectrogram" to fetch the audio and generate a spectrogram. Spectrograms are generated using the default AST FeatureExtractor provided with the original model.

- Now you can play the sound if you'd like.

- Run the audio through the fine-tuned model.
    - Choose how many frequency slices and time slices you'd like to analyze. The defaults are a good start.
    - Click "Classify full audio and all sub-slices".
    - You can now see the fine-tuned model's predicted class for that audio, as well as the "most important" time and frequency slice to that prediction.
    - You can also play the audio matching the "most important" time and frequency slice to see what sounds helped the classification the most.

## What is the "most important" slice?

It's the particular time and/or frequency slice that, if removed, causes the greatest predicted probability drop away from the current audio clip's classification. "Most important" is not the best term here, since it implies that each slice works independently, but you can find some interesting things here. Do dog barks get identified at higher or lower frequencies? Is a "children playing" clip more reliant on the beginning or ending of a kid's playground scream?

## How is the "most important" slice calculated?

Brute force, basically. For example, for a 3-frequency, 3-time slice, the original spectrogram is turned into nine spectrograms of the same size as the original. Each one, however, has a 1/9 rectangular portion (the blank spot) reduced to the mean value of all amplitude levels in the entire dataset (which is 0.4670). Then each new spectrogram is run through the model. After all are run through, the spectrogram with the blank spot that causes the largest drop in probability for the model's classification is computed. Then the location of that blank spot is used to compute the spectrogram approximation that's displayed. It's also used to filter the original audio by time and frequency to allow listening to what was in that particularly importand blank spot.

## Why?

Transformer interpretability is difficult and just plain interesting, with audio transformers even more so. I wanted to see if it was possible to isolate an important portion of an audio signal by both time and frequency to a particular classification task, and more importantly, to _listen_ to it.

This general technique seems worth looking into as a method of data augmentation for small datasets. Making and mixing a set of audio clip portions that represent strong examples of particular classes might be a way to generate more training data.

