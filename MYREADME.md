# Torchtune Usage

## Installation
```
git clone https://github.com/hishab-nlp/torchtune.git
cd torchtune
pip install -e .
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

## Checkpoint conversion

To convert torchtune checkpoint into huggingface checkpoint, run the following script:

- LORA checkpoint conversation

```py
import os
import torch
from torchtune.models.convert_weights import tune_to_peft_adapter_weights


pt_weight_path = "checkpoint_path/adapter.pt"
peft_weight_dir = "lora-peft-output"
os.makedirs(peft_weight_dir, exist_ok=True)

lora_weights = torch.load(pt_weight_path)
lora_converted_weights_peft = tune_to_peft_adapter_weights(lora_weights)
torch.save(lora_converted_weights_peft, os.path.join(peft_weight_dir,"adapter_model.bin"))
# you need to copy the adapater_config.json inside this peft_weight_dir to use this adapter.
```

- Prepare a config `adapter_config.json` with following information according to the training yaml

```json
{
    "r": 32,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj"
    ],
    "peft_type": "LORA"
}
```

NB: If the checkpoint already in Huggingface style then we don't need to do this.

## Inference with trained model

