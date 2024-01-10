import warnings
import logging
import evaluate
import numpy as np
from datasets import Audio, load_dataset
from transformers import (
    ASTFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)
import wandb


logging.basicConfig(
    format="%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel("CRITICAL")
# warnings.filterwarnings("ignore")

# Filter out the specific warning
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly",
)
warnings.filterwarnings(
    "ignore",
    message="promote has been superseded by mode='default'.",
)


def freeze_layers(model, N):
    for name, param in model.named_parameters():
        if len(name.split(".")) >= 4:
            if name.split('.')[3] in [f"{i}" for i in range(N)]:
                param.requires_grad = False


def get_train_val_datasets(dataset, fold=5, fraction=.1):
    train_dataset = dataset.filter(lambda example: example["fold"] != fold)
    val_dataset = dataset.filter(lambda example: example["fold"] == fold)

    # train_dataset = dataset.filter(lambda example: example["slice_file_name"] == "100852-0-0-1.wav")
    # val_dataset = dataset.filter(lambda example: example["slice_file_name"] == "100852-0-0-1.wav")

    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=41)
    val_dataset = val_dataset.shuffle(seed=41)

    # Determine subset sizes
    train_subset_size = int(len(train_dataset) * fraction)
    val_subset_size = int(len(val_dataset) * fraction)

    # Take subsets
    train_subset = train_dataset.select(range(train_subset_size))
    val_subset = val_dataset.select(range(val_subset_size))

    train_subset = train_subset.cast_column("audio", Audio(sampling_rate=16000))
    train_subset = train_subset.rename_column("classID", "labels")

    val_subset = val_subset.cast_column("audio", Audio(sampling_rate=16000))
    val_subset = val_subset.rename_column("classID", "labels")

    train_subset = train_subset.map(
        preprocess_function, batch_size=64, batched=True, num_proc=8, keep_in_memory=True
    )
    val_subset = val_subset.map(
        preprocess_function, batch_size=64, batched=True, num_proc=8, keep_in_memory=True
    )
    return train_subset, val_subset


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        padding="max_length",
        return_tensors="pt",
    )
    return inputs


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(
        predictions=predictions,
        references=labels,
    )


config = {
    "output_dir": "training-0",
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 16,
    "tf32": True,
    "per_device_train_batch_size": 12,
    "logging_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "num_train_epochs": 12,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "constant",
    "report_to": "none",
    "dataloader_num_workers": 8,
    "torch_compile": False,
    "push_to_hub": True,
    "hub_private_repo": True,
    "hub_model_id": "danavery/ast-finetune-urbansound8k"
}
wandb_run = True
wandb_config = config.copy()
freeze_layers_n = 0
wandb_config["frozen_layers"] = freeze_layers_n

if wandb_run:
    config["report_to"] = "wandb"
    wandb.init(project="finetune", config=wandb_config)

for val_fold in range(1, 11):
    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10,
        ignore_mismatched_sizes=True,
    )

    freeze_layers(model, freeze_layers_n)

    dataset = load_dataset("danavery/urbansound8k")["train"]
    feature_extractor = ASTFeatureExtractor()

    train_dataset, val_dataset = get_train_val_datasets(dataset, fraction=.3, fold=val_fold)

    print(model.num_parameters())
    training_args = TrainingArguments(**config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
# model.push_to_hub("danavery/ast-finetune-urbansound8k")
# with torch.no_grad():
#     outputs = model(input_values)

# predicted_class_idx = outputs.logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
