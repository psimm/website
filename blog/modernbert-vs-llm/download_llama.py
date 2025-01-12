# https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/download_llama.py

# ---
# args: ["--force-download"]
# ---

import modal

MODELS_DIR = "/llamas"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

volume = modal.Volume.from_name("llamas", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
