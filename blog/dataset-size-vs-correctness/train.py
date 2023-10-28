import time
from functools import partial
from itertools import product
from random import sample, seed
from typing import Tuple

import modal
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

stub = modal.Stub(
    "correctness-vs-size",
    image=modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "transformers[torch]",
        "datasets",
        "wandb",
        "scikit-learn",  # for metrics
    ),
)

# Create volume to share model and datasets between experiments
stub.volume = modal.Volume.new()


def subsample_hf_dataset(dataset: Dataset, max_size: int):
    # Shuffle dataset
    dataset = dataset.shuffle(seed=42)

    # Separate datasets with labels 0 and 1
    dataset_label_0 = dataset.filter(lambda example: example["label"] == 0)
    dataset_label_1 = dataset.filter(lambda example: example["label"] == 1)

    # Subsample datasets
    subsampled_dataset_label_0 = dataset_label_0.select(range(max_size // 2))
    subsampled_dataset_label_1 = dataset_label_1.select(range(max_size // 2))

    # Concatenate subsampled datasets
    return concatenate_datasets(
        [subsampled_dataset_label_0, subsampled_dataset_label_1]
    )


def load_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


def tokenize_dataset(tokenizer, dataset: Dataset):
    partial_preprocess = partial(preprocess_function, tokenizer)
    return dataset.map(partial_preprocess, batched=True)


def load_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)


def flip_labels(dataset: Dataset, noise_level: float):
    # make the operation deterministic
    seed(42)

    # get number of labels to flip
    n = int(len(dataset) * noise_level)
    n_by_class = n // 2

    # get indices of labels to flip
    neg_indices = [i for i, example in enumerate(dataset) if example["label"] == 0]
    pos_indices = [i for i, example in enumerate(dataset) if example["label"] == 1]

    selected_neg_indices = sample(neg_indices, n_by_class)
    selected_pos_indices = sample(pos_indices, n_by_class)

    # combine indices
    indices_to_flip = selected_neg_indices + selected_pos_indices

    # function to apply to flip the labels
    def flip_labels_function(example, idx: int):
        # flip the label if index is in the selected indices
        # this is not the fastest way to do this, but it's easy to understand
        if idx in indices_to_flip:
            example["label"] = 1 if example["label"] == 0 else 0
        return example

    # apply function to flip the labels
    return dataset.map(flip_labels_function, with_indices=True)


train_args = TrainingArguments(
    learning_rate=2e-5,  # how fast the model learns
    per_device_train_batch_size=16,  # how many training examples are processed at once
    per_device_eval_batch_size=16,  # how many test examples are processed at once
    num_train_epochs=2,  # how many times the model sees the training data
    weight_decay=0.01,  # how much the model is penalized for being complex
    output_dir="./results",
)


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    return {"accuracy": accuracy}


@stub.function(
    concurrency_limit=10,  # max GPU concurrency allowed in free tier
    volumes={"/root/experiment": stub.volume},
    gpu="A10G",
    secret=modal.Secret.from_name("wandb"),
    timeout=15 * 60,
)
def run_experiment(
    train_size: int,
    noise_level: float,
):
    # Load model and datasets
    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained("/root/experiment/model")
    model.to("cuda")

    print("Loading datasets")
    tokenized_train = Dataset.from_parquet("/root/experiment/tokenized_train.parquet")
    tokenized_test = Dataset.from_parquet("/root/experiment/tokenized_test.parquet")

    # subsample and flip labels in training dataset
    tokenized_train_sub = subsample_hf_dataset(tokenized_train, train_size)
    flipped_tokenized_train_sub = flip_labels(tokenized_train_sub, noise_level)

    tokenizer = load_tokenizer()
    data_collator = load_collator(tokenizer)

    # Training
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=flipped_tokenized_train_sub,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    import wandb

    wandb.init(project="correctness-vs-size")

    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start

    evaluation = trainer.evaluate()

    evaluation.update(
        {
            "train_size": train_size,
            "test_size": trainer.eval_dataset.num_rows,
            "noise_level": noise_level,
            "train_time": train_time,
        }
    )

    wandb.log(evaluation)

    wandb.finish()

    return evaluation


def load_and_tokenize_dataset() -> Tuple[Dataset, Dataset]:
    imdb = load_dataset("imdb")
    tokenizer = load_tokenizer()

    tokenized_train = tokenize_dataset(tokenizer, imdb["train"])
    tokenized_test = tokenize_dataset(tokenizer, imdb["test"])

    # Convert to pandas to avoid caching issues
    return tokenized_train, tokenized_test


@stub.function(volumes={"/root/experiment": stub.volume}, timeout=24 * 60 * 60)
def run_experiment_set(train_sizes, noise_levels, test_size) -> list:
    # Do steps that are common to all experiments
    tokenized_train, tokenized_test = load_and_tokenize_dataset()

    # Subsample test dataset to decrease evaluation time
    if not test_size == len(tokenized_test):
        tokenized_test = subsample_hf_dataset(tokenized_test, test_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Share model and tokenized datasets between experiments
    model.save_pretrained("/root/experiment/model")
    tokenized_train.to_parquet("/root/experiment/tokenized_train.parquet")
    tokenized_test.to_parquet("/root/experiment/tokenized_test.parquet")

    stub.volume.commit()  # persist changes

    print("Saved model and datasets to volume")

    combinations = list(product(train_sizes, noise_levels))
    print(f"Number of combinations: {len(combinations)}")

    results = []

    for evaluation in run_experiment.starmap(combinations):
        results.append(evaluation)

    return results


@stub.local_entrypoint()
def main():
    # Define experiment parameters
    train_sizes = np.arange(1000, 15001, 1000)
    noise_levels = np.arange(0, 0.26, 0.025)
    test_size = 10000

    # Run experiments
    results = run_experiment_set.remote(train_sizes, noise_levels, test_size)

    # Write results to file
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_from_modal.csv")
