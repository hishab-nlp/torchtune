# Torchtune Usage

## Installation
```
git install git+https://github.com/pytorch/torchtune.git
```

## Dataset
- Dataset should be in huggingface datasets style
- Each row or JSON line should be in following format

```JSON
{
    "messages": [
        {"role": "system", "content": "prompt content"},
        {"role": "user", "content": "user message 1"},
        {"role": "assistant", "content": "assistant response 1"},
        {"role": "user", "content": "user message 2"},
        {"role": "assistant", "content": "assistant response 2"},
        {"role": "user", "content": "user message 3"},
        {"role": "assistant", "content": "assistant response 3"}
    ]
}
```

## Config preparation
If we want to customize the existing config [recipes/configs](recipes/configs) here is following details.

Our `llama-3-70b-instruct` LORA fine-tuning config examples will be found [here](recipes/configs/hishab_custom).


## Training
- Trianing on single device

```
tune run lora_finetune_single_device --config "path/to/config.yaml"
```

- Training in distributed device

```
tune run lora_finetune_distributed --config "path/to/config.yaml"

```
