from enum import Enum
from pydantic import BaseModel

import outlines
import torch

import modal

image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install("torch", "outlines", "pydantic", "transformers", "datasets", "accelerate")

stub = modal.Stub(image=image)


class Polarity(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class Importance(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Aspect(BaseModel):
    aspect: str
    polarity: Polarity
    importance: Importance


class Absa(BaseModel):
    aspects: list[Aspect]


@stub.function(secrets=[modal.Secret.from_dotenv()], gpu=modal.gpu.A100(count=1))
def generate_with_model(prompt: str):
    device = torch.device("cuda")

    model = outlines.models.transformers(
        "mistralai/Mistral-7B-Instruct-v0.2", device=device
    )
    generator = outlines.generate.json(model, Absa)

    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    analysis = generator(prompt, rng=rng)

    return repr(analysis)


@stub.local_entrypoint()
def main():
    prompt = "Analyze this review: The food was tasty but the service was slow"
    output = generate_with_model.remote(prompt)

    print(output)
