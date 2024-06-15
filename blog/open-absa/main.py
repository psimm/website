# %%
import json
import os
from typing import List

import datasets
import polars as pl
import torch
import wandb
import xmltodict
from modal import Image, Secret, App, gpu
from peft import LoraConfig  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTTrainer

GPU_CONFIG = gpu.A100(count=1, memory=40)  # type: ignore
HUGGINGFACE_USER = "psimm"

DO_TRAIN = False
DO_EVALUATE = True

image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install(
    "transformers",
    "datasets",
    "polars",
    "torch",
    "trl",
    "peft",
    "xmltodict",
    "wandb",
)

app = App()


def read_semeval_xml(path: str) -> pl.DataFrame:
    with open(path, "rb") as f:
        parsed_dict = xmltodict.parse(f.read())

    sentences_list = parsed_dict["sentences"]["sentence"]

    for sentence in sentences_list:
        sentence["sentence_id"] = sentence["@id"]
        sentence["aspect_terms"] = []

        if "aspectTerms" in sentence:
            aspect_terms = sentence["aspectTerms"]["aspectTerm"]
            if isinstance(aspect_terms, dict):
                aspect_terms = [aspect_terms]

            for aspect_term in aspect_terms:
                aterm = {
                    "to": aspect_term["@to"],
                    "from": aspect_term["@from"],
                    "term": aspect_term["@term"],
                    "polarity": aspect_term["@polarity"],
                }
                sentence["aspect_terms"].append(aterm)

    df = pl.DataFrame(sentences_list).select("sentence_id", "text", "aspect_terms")

    return df


def aspect_terms_to_output(aspect_terms: dict) -> str:
    return json.dumps(
        [
            {
                "term": aterm["term"],
                "polarity": aterm["polarity"],
            }
            for aterm in aspect_terms
        ]
    )


# Use LoRA to train an adapter instead of the full model
# https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
# https://medium.com/@manyi.yim/more-about-loraconfig-from-peft-581cf54643db
peft_config = LoraConfig(
    r=16,  # Size of the adapter, larger = better, but slower
    lora_alpha=32,  # Raschka: a good heuristic is to set lora_alpha to 2 * r
    lora_dropout=0.05,  # Good against overfitting
    bias="none",
    task_type="CAUSAL_LM",  # next token prediction
)


def formatting_prompts_func(example: dict) -> List[str]:
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def build_hf_dataset(data_train: pl.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "instruction": data_train["text"].to_list(),
            "output": [
                aspect_terms_to_output(a) for a in data_train["aspect_terms"].to_list()
            ],
        }
    )


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=3600,
    secrets=[Secret.from_dotenv()],
)
def train_model(
    repo_id: str,
    data: pl.DataFrame,
    peft_config: LoraConfig,
) -> None:

    hf_token = os.environ["HF_TOKEN"]
    assert hf_token is not None, "HF_TOKEN environment variable is not set"

    wandb_api_key = os.environ["WANDB_API_KEY"]
    assert wandb_api_key is not None, "WANDB_API_KEY environment variable is not set"

    dataset = build_hf_dataset(data)

    # Load model from HuggingFace model hub
    model = AutoModelForCausalLM.from_pretrained(repo_id).to("cuda")

    # https://huggingface.co/docs/trl/main/en/sft_trainer
    # Train on both questions and answers
    # Training on only answers gave worse results
    # https://github.com/huggingface/trl/issues/632
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,  # type: ignore
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        max_seq_length=1024,
    )

    trainer.train()  # type: ignore

    wandb.finish()

    # Push LoRA adapter to HuggingFace model hub
    model_last_name = repo_id.split("/")[-1]
    upload_repo_id = f"{HUGGINGFACE_USER}/{model_last_name}-semeval2014"

    peft_model = trainer.model

    peft_model.push_to_hub(repo_id=upload_repo_id, private=True, token=hf_token)  # type: ignore


def batch_infer(model: AutoModelForCausalLM, tokenizer, texts: List[str]):
    device = torch.device("cuda")
    model.to(device)  # type: ignore

    answers = []

    # https://huggingface.co/docs/transformers/v4.39.3/en/generation_strategies
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=128,
        temperature=0.01,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )

        inputs.to(device)

        outputs = model.generate(  # type: ignore
            **inputs,
            generation_config=generation_config,
        )

        # Remove the input text from the output
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]

        outputs_decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answers.append(outputs_decoded)

    return answers


def absa_dict_to_str(dict) -> str:
    return json.dumps(dict)


def evaluate_example(expected: list, predicted: list) -> dict[str, int]:
    # Turn the dicts into strings so they're hashable
    expected_terms = set([absa_dict_to_str(x) for x in expected])
    predicted_terms = set([absa_dict_to_str(x) for x in predicted])

    evaluation = {
        "tp": len(expected_terms & predicted_terms),
        "fp": len(predicted_terms - expected_terms),
        "fn": len(expected_terms - predicted_terms),
    }

    # Handle the case where there are no aspects in the target or prediction
    # This means the model correctly predicted "no aspects"
    # InstructABSA handles it this way
    if len(expected_terms) == 0 and len(predicted_terms) == 0:
        evaluation["tp"] = 1

    return evaluation


@app.function(image=image, gpu=GPU_CONFIG, timeout=3600, secrets=[Secret.from_dotenv()])
def evaluate_model(data_test: pl.DataFrame, repo_id: str) -> dict:
    hf_token = os.environ["HF_TOKEN"]
    assert hf_token is not None, "HF_TOKEN environment variable is not set"

    # Load my trained model
    # The adapter and the base model are loaded together
    model = AutoModelForCausalLM.from_pretrained(repo_id, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # Get predictions
    texts = data_test["text"].to_list()
    texts = [f"### Question: {t}\n ### Answer: " for t in texts]

    answers = batch_infer(model, tokenizer, texts)

    # Parse answers
    expectations = data_test["aspect_terms"].to_list()
    predictions = []
    num_parsing_errors = 0

    for answer in answers:
        try:
            predictions.append(json.loads(answer))
        except json.JSONDecodeError:
            predictions.append([])  # treat as no aspects
            num_parsing_errors += 1

    # Evaluate answers
    evaluations = [
        evaluate_example(expected, predicted)
        for expected, predicted in zip(expectations, predictions)
    ]

    # Calculate metrics
    metrics_df = (
        pl.DataFrame(evaluations)
        .sum()
        .select(
            model=pl.lit(repo_id),
            parsing_errors=pl.lit(num_parsing_errors),
            precision=pl.col("tp") / (pl.col("tp") + pl.col("fp")),
            recall=pl.col("tp") / (pl.col("tp") + pl.col("fn")),
        )
        .with_columns(
            f1=2
            * (pl.col("precision") * pl.col("recall"))
            / (pl.col("precision") + pl.col("recall")),
        )
    )

    return metrics_df.to_dicts()[0]


# Define list of models to train and evaluate
# https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
repo_ids = ["meta-llama/Meta-Llama-3-8B"]


@app.local_entrypoint()
def main():

    data_files = [
        {"path": "data/laptops_train.xml", "domain": "laptops", "split": "train"},
        {"path": "data/laptops_test.xml", "domain": "laptops", "split": "test"},
        {
            "path": "data/restaurants_train.xml",
            "domain": "restaurants",
            "split": "train",
        },
        {"path": "data/restaurants_test.xml", "domain": "restaurants", "split": "test"},
    ]

    data_raw = pl.concat(
        [
            read_semeval_xml(file["path"]).with_columns(
                domain=pl.lit(file["domain"]), split=pl.lit(file["split"])
            )
            for file in data_files
        ]
    )

    print(f"Loaded {len(data_raw)} examples from SemEval 2014 dataset")

    # Remove sentences that have the conflict polarity label
    rm_sentence_ids = (
        data_raw.explode("aspect_terms")
        .unnest("aspect_terms")
        .filter(polarity="conflict")
        .get_column("sentence_id")
        .unique()
    )

    data = data_raw.filter(~pl.col("sentence_id").is_in(rm_sentence_ids))

    print(f"Removed {len(rm_sentence_ids)} sentences with conflict polarity")

    data_train = data.filter(split="train")
    data_test = data.filter(split="test")

    print("Finished preparing dataset")

    if DO_TRAIN:
        for i, repo_id in enumerate(repo_ids):
            print(f"Training model {i+1}/{len(repo_ids)}: {repo_id}")
            train_model.remote(repo_id, data_train, peft_config)

    if DO_EVALUATE:
        metrics = [
            evaluate_model.remote(data_test, repo_id)
            for repo_id in [
                f"{HUGGINGFACE_USER}/Meta-Llama-3-8B",
            ]
        ]

        metrics_df = pl.DataFrame(metrics)
        print(metrics_df)


# Consider vllm: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
