import json
import os
import subprocess
import time
from typing import Literal

import modal

# See CUDA guide for more details: https://modal.com/docs/guide/cuda
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchao==0.6.1",
        "wandb==0.18.5",
        "lm_eval==0.4.5",
        "vllm==0.6.6.post1",
        "scikit-learn==1.6.1",
    )
    .run_commands(
        # Ray and torchtune both use the name "tune" for their CLI
        # Move Ray's tune to tune-ray to preserve it if needed
        "mv /usr/local/bin/tune /usr/local/bin/tune-ray",
    )
    .pip_install("torchtune==0.5.0")
    .pip_install("ipython")  # TODO: remove
)
app = modal.App(image=image)

vol = modal.Volume.from_name("torchtune-checkpoints", create_if_missing=True)


@app.function(
    volumes={"/checkpoints": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=900,
)
def download(model: str):
    checkpoint_dir = f"/checkpoints/{model}"

    if not os.path.exists(checkpoint_dir):
        print(f"Downloading {model} to {checkpoint_dir}")
        subprocess.run(["tune", "download", model, "--output-dir", checkpoint_dir])
    else:
        print(f"{model} already exists in {checkpoint_dir}")


@app.function(
    gpu="a10g",
    volumes={"/checkpoints": vol},
    secrets=[modal.Secret.from_name("wandb")],
    mounts=[
        modal.Mount.from_local_dir("configs", remote_path="/tmp/configs"),
        modal.Mount.from_local_dir("data", remote_path="/data"),
    ],
    timeout=3 * 3600,
)
def train(config: str):
    import torch

    assert torch.cuda.is_available(), "CUDA is not available"

    subprocess.run(
        [
            "tune",
            "run",
            "lora_finetune_single_device",
            "--config",
            f"/tmp/configs/{config}",
        ]
    )


@app.function(
    gpu="a10g",
    volumes={"/checkpoints": vol},
)
def inference(
    model: str,
    messages: list[list[dict[str, str]]],
    lora_path: str | None = None,
) -> tuple[list[str], dict[str, float]]:
    """
    Run inference on a list of prompts.

    Args:
        model: Path to the model checkpoint
        prompts: List of strings to use as inputs
        lora_path: Path to the LoRA adapter to use for inference

    Returns:
        results: The generated text for each prompt
        stats: Inference statistics as a dictionary
    """
    from vllm import LLM, SamplingParams

    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    # https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams
    # https://docs.vllm.ai/en/stable/models/generative_models.html#llm-generate
    llm = LLM(
        model=model,
        enable_lora=lora_path is not None,
        max_model_len=1000,
    )
    sampling_params = SamplingParams(temperature=0.0)

    generate_kwargs = {
        "messages": messages,
        "sampling_params": sampling_params,
    }

    if lora_path is not None:
        from vllm.lora.request import LoRARequest

        print(f"Using LoRA adapter {lora_path}")
        print(f"Adapter files present: {os.listdir(lora_path)}")
        # LoRARequest takes (name, id, path) as arguments
        generate_kwargs["lora_request"] = LoRARequest(
            "adapter",  # Human readable name
            0,  # Unique integer ID
            lora_path,  # Path to adapter
        )

    start_time = time.time()
    outputs = llm.chat(**generate_kwargs)
    end_time = time.time()
    duration = end_time - start_time

    results = [output.outputs[0].text for output in outputs]

    stats = {
        "duration": duration,
        "examples_per_sec": len(messages) / duration,
    }

    return results, stats


@app.function(
    gpu="a10g",
    volumes={"/checkpoints": vol},
    mounts=[modal.Mount.from_local_dir("data", remote_path="/data")],
    secrets=[modal.Secret.from_name("wandb")],
)
def evaluate(
    model: str,
    lora_path: str | None = None,
    split: Literal["train", "test", "valid"] = "valid",
    wandb_run_id: str | None = None,
    print_first_n_parsing_errors: int = 5,
):
    import wandb
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    assert split in ["train", "test", "valid"], "Invalid split"

    with open(f"/data/{split}.json", "r") as f:
        chats = json.load(f)

    # vLLM expects messages in OpenAI format
    messages = [
        [{"role": "user", "content": chat["dialogue"][0]["value"]}] for chat in chats
    ]

    print(f"Running inference on {len(messages)} messages")

    results, stats = inference.remote(
        model=model, lora_path=lora_path, messages=messages
    )

    metrics = {
        f"{split}/examples_per_sec": stats["examples_per_sec"],
        f"{split}/duration": stats["duration"],
    }

    parsing_errors = 0
    predictions = []
    for result in results:
        try:
            predictions.append(int(result))
        except ValueError:
            if parsing_errors < print_first_n_parsing_errors:
                print(f"Error parsing result: '{result}' as int")
            parsing_errors += 1
            predictions.append(0)

    expected_labels = [int(chat["dialogue"][1]["value"]) for chat in chats]

    metrics[f"{split}/accuracy"] = accuracy_score(expected_labels, predictions)
    metrics[f"{split}/recall"] = recall_score(expected_labels, predictions)
    metrics[f"{split}/precision"] = precision_score(expected_labels, predictions)

    metrics[f"{split}/parsing_errors"] = parsing_errors

    print(metrics)

    if wandb_run_id:
        wandb.init(
            project="modernbert-vs-llm",
            id=wandb_run_id,
            resume="allow",
        )
        wandb.log(metrics)
        wandb.finish()


@app.local_entrypoint()
def main(
    download_model: bool = False,
    run_training: bool = False,
    run_evaluation: bool = False,
    evaluation_split: str = "valid",
):
    """
    Run the specified steps of the Llama fine-tuning pipeline.

    Args:
        download_model: Whether to download the base model
        run_training: Whether to run the LoRA fine-tuning
        run_evaluation: Whether to evaluate the fine-tuned model
        evaluation_split: The split to use for evaluation. One of "train", "test", "valid"
    """

    if download_model:
        download.remote("meta-llama/Llama-3.2-3B-Instruct")
    if run_training:
        train.remote("3B_lora_single_device.yaml")
    if run_evaluation:
        # vLLM doesn't use the LoRA adapter at inference time unless
        # it's merged into the base model
        # https://github.com/vllm-project/vllm/issues/6250
        evaluate.remote(
            model="/checkpoints/trained/meta-llama/Llama-3.2-3B-Instruct/lora_single_device/epoch_3",
            wandb_run_id="ve2pe9xi",
            split=evaluation_split,
        )
