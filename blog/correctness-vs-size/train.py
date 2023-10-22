import time
from functools import partial
from itertools import product
from random import sample, seed

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


def load_model(device: str = "cuda"):
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(
        device
    )  # use GPU on Modal

    return model


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


stub = modal.Stub(
    "correctness-vs-size",
    image=modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "transformers[torch]",
        "datasets",
        "scikit-learn",  # for metrics
    ),
)


@stub.cls(gpu="A10G")
class Experiment:
    def __init__(
        self,
        train_size: int,
        test_size: int,
        noise_level: float,
        train_args,
        compute_metrics,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.noise_level = noise_level
        self.compute_metrics = compute_metrics
        self.train_args = train_args
        self.model = load_model("cuda")
        self.tokenizer = load_tokenizer()
        self.data_collator = load_collator(self.tokenizer)

    def load_data(self):
        # load data
        imdb = load_dataset("imdb")

        # tokenize texts
        tokenized_train = tokenize_dataset(self.tokenizer, imdb["train"])
        tokenized_test = tokenize_dataset(self.tokenizer, imdb["test"])

        # subsample and flip labels
        self.train_sub = subsample_hf_dataset(tokenized_train, self.train_size)
        self.train_sub = flip_labels(self.train_sub, self.noise_level)
        self.test_sub = subsample_hf_dataset(tokenized_test, self.test_size)

    def setup_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_sub,
            eval_dataset=self.test_sub,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def train_and_evaluate(self):
        train_start = time.time()
        self.trainer.train()
        train_time = time.time() - train_start

        evaluation = self.trainer.evaluate()

        evaluation.update(
            {
                "train_size": self.train_size,
                "test_size": self.test_size,
                "noise_level": self.noise_level,
                "train_time": train_time,
            }
        )

        return evaluation

    @modal.method()
    def run(self):
        self.load_data()
        self.setup_trainer()
        evaluation = self.train_and_evaluate()

        return evaluation


@stub.function(concurrency_limit=5)
def run_experiment(train_size: int, noise_level: float):
    experiment = Experiment(
        train_size=train_size,
        test_size=1000,
        noise_level=noise_level,
        train_args=train_args,
        compute_metrics=compute_metrics,
    )
    evaluation = experiment.run.remote()

    print(evaluation)
    return evaluation


@stub.local_entrypoint()
def main():
    train_sizes = np.arange(1000, 5001, 1000)
    noise_levels = np.arange(0, 0.25, 0.025)

    combinations = list(product(train_sizes, noise_levels))
    print(f"Number of combinations: {len(combinations)}")

    results = []

    for evaluation in run_experiment.starmap(combinations):
        results.append(evaluation)

    # Write results to file
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_from_modal.csv")
