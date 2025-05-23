# Running the experiments

## Requirements

- [Modal](https://modal.com) account and the Modal CLI installed
- [Weights and Biases](https://wandb.ai/site) (W&B) account. This is optional, but it's useful for tracking the results of the experiments.
- [HuggingFace](https://huggingface.co) account with access to the gated [Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model
- Python >= 3.10 installed and the packages listed in `requirements.txt` installed
- Secrets for HuggingFace and W&B set up in Modal
- Environment variable `FASTAPI_VLLM_TOKEN` set to a value of your choice. This is used to secure the vLLM server.

## Preparing the dataset

Run `python prep.py` to load the dataset and split it into training, validation and test sets. The results will be saved in the `data` directory in sharegpt JSON format.

## Experiment setups

The setups can be run independently and in any order. Only setup 2 requires a running vLLM server, the other two don't.

### Setup 1: Fine-tuning ModernBERT

Use the `modernbert.py` script to fine-tune ModernBERT on Modal. It runs in two steps:

1. `modal run modernbert.py::train_modernbert` to train the model. It will be uploaded to HuggingFace.
2. `modal run modernbert.py::test_modernbert` to run inference on the test set.

Both commands accept additional arguments to specify the model size. Check `modernbert.py` for more details.

### Setup 2: Few-shot learning with DSPy

This requires a running vLLM server. Use `modal deploy vllm_inference.py` to start it. Then run the `fewshot_dspy.py` script to optimize a prompt and then test it. Don't forget to shut down the vLLM server after you're done using `modal app stop <app_id>`. You can find the app id with `modal app list`.

### Setup 3: Fine-tuning Llama 3.2-3B

This setup has 3 steps: download a base model, fine-tune it on the dataset and run evaluation. The commands are:

1. `modal run finetune_llama.py --download-model` to download the base model from HuggingFace into a Modal volume.
2. `modal run finetune_llama.py --run-training` to train the model and save the updated model to the same volume.
3. `modal run finetune_llama.py --run-evaluation --evaluation-split <split>` to run evaluation. Use either `test` or `valid` as the split. Results are uploaded to W&B.

They can also be combined in a single command: `modal run finetune_llama.py --download-model --run-training --run-evaluation --evaluation-split <split>`. Check the arguments in the `finetune_llama.py` script for more details.

Use `modal run finetune_llama.py` to fine-tune a Llama 3.2-3B model on Modal. The command accepts flags for 3 steps: download, run training and run evaluation. The model is downloaded from HuggingFace, trained using Modal and uploaded to a Modal volume. The evaluation results are uploaded to W&B.

## Getting help

Please open an issue in this Github repository or [contact](https://simmering.dev/about) me if you need help running the experiments yourself.
