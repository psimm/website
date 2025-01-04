import json
import math
import os

import polars as pl
import datasets


def prep_dataset(
    train_pct: float = 0.6,
    test_pct: float = 0.2,
    valid_pct: float = 0.2,
    seed: int = 42,
) -> datasets.arrow_dataset.Dataset:
    """
    Load the ADE Corpus v2 classification dataset from Hugging Face
    and prepare it for training and inference.
    """

    # There is only a train split in the dataset on Hugging Face
    dataset = datasets.load_dataset(
        path="ade_corpus_v2", name="Ade_corpus_v2_classification", split="train"
    )

    df_raw = dataset.to_polars()

    # Deduplicate dataset based on text column
    n_before = df_raw.height
    df = df_raw.unique("text")
    n_after = df.height

    print(f"Deduplicated dataset from {n_before} to {n_after} rows.")

    # Check distribution of labels
    df["label"].value_counts()

    # Subsample to balance classes
    smallest_class_n = df["label"].value_counts().min()["count"][0]

    df_balanced = df.filter(
        pl.int_range(pl.len()).shuffle().over("label") < smallest_class_n
    ).sample(fraction=1, seed=seed)

    df_balanced["label"].value_counts()

    print(f"Balanced dataset to {smallest_class_n} examples per label")

    # Train, test, validation split
    # Calculate number of examples per split for each class
    n_per_class = df_balanced.group_by("label").len()["len"][0]
    train_n_per_class = math.floor(train_pct * n_per_class)
    test_n_per_class = math.floor(test_pct * n_per_class)

    # Create stratified splits
    df_split = (
        df_balanced.sample(fraction=1, shuffle=True, seed=seed)
        .with_columns(row_number=pl.int_range(0, pl.len()).over("label"))
        .with_columns(
            split=pl.when(pl.col("row_number") < train_n_per_class)
            .then(pl.lit("train"))
            .when(pl.col("row_number") < (train_n_per_class + test_n_per_class))
            .then(pl.lit("test"))
            .otherwise(pl.lit("valid"))
        )
        .drop("row_number")
    )

    print("Split distribution:")
    print(
        df_split.group_by("split", "label").len(name="examples").sort("split", "label")
    )

    # Turn back into HF dataset format
    splits = {}
    for split_name in ["train", "test", "valid"]:
        split_df = df_split.filter(pl.col("split") == split_name).drop("split")
        splits[split_name] = datasets.Dataset.from_polars(split_df)

    ds = datasets.DatasetDict(splits)

    return ds


def dataset_to_openai_jsonl(
    dataset: datasets.arrow_dataset.Dataset,
    output_path: str,
) -> None:
    # See documentation for openai conversation style
    # https://pytorch.org/torchtune/main/basics/chat_datasets.html#openai

    # Format in openai conversation style
    chats = [
        {
            "messages": [
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": str(row["label"])},
            ]
        }
        for row in dataset
    ]

    with open(output_path, "w") as f:
        for chat in chats:
            f.write(json.dumps(chat) + "\n")


if __name__ == "__main__":
    ds = prep_dataset()
    os.makedirs("data", exist_ok=True)
    dataset_to_openai_jsonl(ds["train"], "data/train.jsonl")
    dataset_to_openai_jsonl(ds["test"], "data/test.jsonl")
    dataset_to_openai_jsonl(ds["valid"], "data/valid.jsonl")
