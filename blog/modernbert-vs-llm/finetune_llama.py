import json
import os
import time
from typing import Literal

import subprocess

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
    )
    .run_commands(
        # Ray and torchtune both use the name "tune" for their CLI
        # Move Ray's tune to tune-ray to preserve it if needed
        "mv /usr/local/bin/tune /usr/local/bin/tune-ray",
    )
    .pip_install("torchtune==0.5.0")
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
    prompts: list[str],
    lora_path: str | None = None,
) -> list[tuple[str, str]]:
    from vllm import LLM, SamplingParams

    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    # https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams
    llm = LLM(
        model=model,
        enable_lora=lora_path is not None,
        max_model_len=1000,
    )
    sampling_params = SamplingParams(temperature=0.0)

    generate_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
    }

    if lora_path is not None:
        # https://docs.vllm.ai/en/latest/models/lora.html
        from vllm.lora.request import LoRARequest

        print(f"Using LoRA adapter {lora_path}")
        generate_kwargs["lora_request"] = LoRARequest(
            lora_name="adapter", lora_path=lora_path, lora_int_id=0
        )

    outputs = llm.generate(**generate_kwargs)

    results = [output.outputs[0].text for output in outputs]

    return results


@app.function(
    gpu="a10g",
    volumes={"/checkpoints": vol},
    mounts=[modal.Mount.from_local_dir("data", remote_path="/data")],
    secrets=[modal.Secret.from_name("wandb")],
)
def evaluate(
    model: str,
    lora_path: str | None = None,
    split: Literal["train", "test", "valid"] = "test",
    wandb_run_id: str | None = None,
    print_first_n_parsing_errors: int = 5,
):
    import wandb

    chats = []
    with open(f"/data/{split}.jsonl", "r") as f:
        for line in f:
            chats.append(json.loads(line))

    prompts = [chat["messages"][0]["content"] for chat in chats]
    expected_labels = [int(chat["messages"][1]["content"]) for chat in chats]

    print(f"Running inference on {len(prompts)} prompts")

    start_time = time.time()
    results = inference.remote(model=model, lora_path=lora_path, prompts=prompts)
    end_time = time.time()
    duration = end_time - start_time
    examples_per_sec = len(prompts) / duration
    metrics = {
        "test/examples_per_sec": examples_per_sec,
        "test/duration": duration,
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

    metrics["test/accuracy"] = sum(
        pred == expected for pred, expected in zip(predictions, expected_labels)
    ) / len(results)

    metrics["test/parsing_errors"] = parsing_errors

    print(metrics)

    if wandb_run_id:
        wandb.init(project="modernbert-vs-llm", id=wandb_run_id, resume="allow")
        wandb.log(metrics)
        wandb.finish()


@app.local_entrypoint()
def main(
    download_model: bool = False,
    run_training: bool = False,
    run_evaluation: bool = False,
):
    """
    Run the specified steps of the Llama fine-tuning pipeline.

    Args:
        download_model: Whether to download the base model
        run_training: Whether to run the LoRA fine-tuning
        run_evaluation: Whether to evaluate the fine-tuned model
    """
    if download_model:
        download.remote("meta-llama/Llama-3.2-3B-Instruct")
    if run_training:
        train.remote("3B_lora_single_device.yaml")
    if run_evaluation:
        evaluate.remote(
            model="/checkpoints/meta-llama/Llama-3.2-3B-Instruct",
            lora_path="/checkpoints/trained/meta-llama/Llama-3.2-3B-Instruct/lora_single_device",
            wandb_run_id="fblv2wis",
        )
