# Implement few-shot learning with Llama 3.2-3B and DSPy
# Requires a vLLM server with the model running
# Deploy vllm_inference.py on Modal
# https://modal.com/docs/examples/vllm_inference

import os
from typing import Literal

import dspy
import wandb
from dotenv import load_dotenv
from download_llama import MODEL_NAME
from prep import prep_dataset

load_dotenv()

NUM_THREADS = 25

# Connect to the vLLM server
# dspy uses litellm under the hood
# https://docs.litellm.ai/docs/providers/vllm
MODEL_ID = MODEL_NAME.split("/")[1]
model = "hosted_vllm/" + MODEL_ID
api_base = "https://psimm--example-vllm-openai-compatible-serve.modal.run/v1/"
api_key = os.getenv("FASTAPI_VLLM_TOKEN")

lm = dspy.LM(
    api_base=api_base,
    api_key=api_key,
    model=model,
    temperature=0.0,  # best for structured outputs
    cache=True,
    max_tokens=250,
)


dspy.configure(lm=lm)
lm("Say this is a test!", temperature=0.7)

# Turn examples into DSPy example class instances
dataset = prep_dataset()


# Add a base prompt to the examples
# This is a bit of a hack to combine DSPy with manual prompting
def to_example(row):
    base_prompt = (
        "Determine if the following sentence is about adverse drug reactions: "
    )

    return dspy.Example(
        text=base_prompt + row["text"],
        label=row["label"],
    ).with_inputs("text")


def split_to_examples(split):
    return [to_example(row) for row in dataset[split].to_polars().to_dicts()]


examples_train = split_to_examples("train")
examples_valid = split_to_examples("valid")
examples_test = split_to_examples("test")


# Signature for the predictor
# This ensures that it will only return either 0 or 1 for each example
class ClassificationSignature(dspy.Signature):
    text: str = dspy.InputField()
    label: Literal[0, 1] = dspy.OutputField()


predictor = dspy.Predict(ClassificationSignature)

# Quick test
predictor(text="I took the pills and now I feel sick.")


# Set up optimizer
def evaluate(
    example: dspy.Example, prediction: dspy.primitives.prediction.Prediction, trace=None
) -> Literal[0, 1]:
    return 1 if prediction.label == example.label else 0


optimizer = dspy.teleprompt.MIPROv2(metric=evaluate, num_threads=NUM_THREADS)

# Select optimal few-shot examples and prompt instructions
optimized_predictor = optimizer.compile(
    student=predictor,
    trainset=examples_train,
    valset=examples_valid,
    minibatch_size=50,  # evaluate changes on a subset of the validation set
    minibatch_full_eval_steps=10,  # evaluate on the full validation set after every 10 steps
    max_labeled_demos=20,  # the number of few-shot examples to use
    max_bootstrapped_demos=5,  # come up with additional illustrative examples
    num_trials=3,  # how many combinations of few-shot examples and prompt instructions to try
    seed=42,  # for reproducibility
    requires_permission_to_run=False,  # skip confirmation dialog
)

optimized_predictor.save("optimized_predictor.json")


# Evaluate on the test set
evaluator = dspy.Evaluate(
    devset=examples_test,
    metric=evaluate,
    display_progress=True,
    num_threads=NUM_THREADS,
)

score = evaluator(optimized_predictor) / 100

print(f"Accuracy: {score}")

# Log to W&B
wandb.init(
    project="modernbert-vs-llm",
    config={
        "method": "few-shot",
        "model": MODEL_NAME,
        "state": optimized_predictor.dump_state(),
    },
)

wandb.log({"accuracy": score})

wandb.finish()
