from peft import LoraConfig, get_peft_model
import torch.nn as nn


def create_lora_vit_model(
    model: nn.Module,
    lora_rank: int,
    lora_initialization_strategy: str,
    lora_target_modules=None,
    lora_alpha: int | None = None,
):
    alpha = lora_alpha if lora_alpha is not None else lora_rank
    if lora_initialization_strategy == "random":
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=alpha,
            init_lora_weights=True,
            target_modules=lora_target_modules,
            modules_to_save=["head"],
        )
    elif lora_initialization_strategy == "dora":
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=alpha,
            use_dora=True,
            target_modules=lora_target_modules,
            modules_to_save=["head"],
        )
    elif lora_initialization_strategy == "rslora":
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=alpha,
            use_rslora=True,
            target_modules=lora_target_modules,
            modules_to_save=["head"],
        )

    peft_vit_classifier = get_peft_model(model, lora_config)
    print(peft_vit_classifier.print_trainable_parameters())
    return peft_vit_classifier
