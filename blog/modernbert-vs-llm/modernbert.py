# Training code for ModernBERT on ADE Classification on Modal
# Adapted from https://www.philschmid.de/fine-tune-modern-bert-in-2025
# and https://github.com/DrChrisLevy/DrChrisLevy.github.io/blob/main/posts/modern_bert/trainer.py

# Usage:
# modal run modernbert.py::train_modernbert --model-id answerdotai/ModernBERT-base
# modal run modernbert.py::test_modernbert --model-id psimm/ModernBERT-base-ade-corpus-v2-classification --wandb-run-id <wandb-run-id>

# modal run modernbert.py::train_modernbert --model-id answerdotai/ModernBERT-large
# modal run modernbert.py::test_modernbert --model-id psimm/ModernBERT-large-ade-corpus-v2-classification --wandb-run-id <wandb-run-id>

import modal

from prep import prep_dataset

# Flash-Attn Image
# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.4.1")
    .pip_install(
        "ninja",  # required to build flash-attn
        "packaging",  # required to build flash-attn
        "wheel",  # required to build flash-attn
        "setuptools<71.0.0",
        "scikit-learn",
        "datasets==3.1.0",
        "accelerate==1.2.1",
        "hf-transfer==0.1.8",
        "polars==1.13.1",
        "wandb",
    )
    .pip_install("flash-attn")
    # Install transformers from github
    .pip_install(
        "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1",
    )
)

app = modal.App(
    image=image,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)


@app.function(image=image)
def tokenize(dataset):
    from transformers import AutoTokenizer

    # Model id to load the tokenizer
    model_id = "google-bert/bert-base-uncased"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512  # longer examples are truncated

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, return_tensors="pt"
        )

    # Tokenize dataset
    raw_dataset = dataset.rename_column("label", "labels")  # to match Trainer
    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

    return tokenized_dataset


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score

    predictions, labels = eval_pred

    # Convert predictions to class labels
    predicted_class = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_class)
    recall = recall_score(labels, predicted_class)
    precision = precision_score(labels, predicted_class)

    # Add binary cross entropy
    bce = log_loss(labels, predictions)
    return {
        "accuracy": float(accuracy),
        "bce": float(bce),
        "recall": float(recall),
        "precision": float(precision),
    }


@app.function(image=image, timeout=60 * 60, gpu="A10G")
def train(tokenized_dataset, model_id: str):
    from huggingface_hub import HfFolder
    from transformers import Trainer, TrainingArguments
    import wandb
    from transformers import AutoModelForSequenceClassification

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    wandb.init(project="modernbert-vs-llm")

    model_name = model_id.split("/")[-1]
    output_dir = f"{model_name}-ade-corpus-v2-classification"

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True,  # bfloat16 training
        optim="adamw_torch_fused",  # improved optimizer
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bce",
        report_to="wandb",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    wandb.finish()


@app.function(image=image, timeout=60 * 60, gpu="A10G")
def test(dataset, model_id: str, wandb_run_id: str = None):
    import time
    from transformers import pipeline
    from transformers import AutoTokenizer
    import numpy as np
    import wandb

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    classifier = pipeline(
        model=model_id,
        task="text-classification",
        device=0,
        tokenizer=tokenizer,
        return_all_scores=True,
    )

    test_set = dataset["test"]

    # Predict and time
    start_time = time.time()
    predictions = classifier(test_set["text"], batch_size=128)
    end_time = time.time()
    duration = end_time - start_time
    examples_per_sec = len(test_set) / duration
    metrics = {
        "test/examples_per_sec": examples_per_sec,
        "test/duration": duration,
    }

    # Evaluate predictions
    labels = test_set["label"]
    predictions_array = np.array(
        [[pred[0]["score"], pred[1]["score"]] for pred in predictions]
    )

    eval_pred = (predictions_array, labels)
    eval_metrics = compute_metrics(eval_pred)
    metrics["test/accuracy"] = eval_metrics["accuracy"]
    metrics["test/bce"] = eval_metrics["bce"]
    metrics["test/recall"] = eval_metrics["recall"]
    metrics["test/precision"] = eval_metrics["precision"]

    if wandb_run_id:
        wandb.init(project="modernbert-vs-llm", id=wandb_run_id, resume="allow")
        wandb.log(metrics)
        wandb.finish()

    return metrics


@app.function(image=image, timeout=60 * 60)
def train_modernbert(model_id: str = "answerdotai/ModernBERT-base"):
    dataset = prep_dataset()
    tokenized_dataset = tokenize.remote(dataset)
    train.remote(tokenized_dataset, model_id)  # model is saved to hub


@app.function(image=image, timeout=60 * 60)
def test_modernbert(wandb_run_id: str, model_id: str = "answerdotai/ModernBERT-base"):
    dataset = prep_dataset()
    metrics = test.remote(dataset, wandb_run_id=wandb_run_id, model_id=model_id)
    print(metrics)
