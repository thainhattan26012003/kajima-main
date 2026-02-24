import shutil
import wandb
import os
from transformers import AutoImageProcessor, AutoModel
from src.model import ViTClassifier, get_target_modules
from peft import LoraConfig, get_peft_model
from src.utils.trainer import Trainer
from typing import Any
from argparse import ArgumentParser
import json
import torch


def agent_sweep(
    config: dict[str, Any],
    sweep_id: str,
    train_dataset_dir: str,
    test_dataset_dir: str,
    base_model_path: str,
    output_dir: str = None,
    device: str = "cpu",
):
    api = wandb.Api()

    with wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        config=config,
        tags=["2025-3-24-base-finetuning"],
    ) as run:
        image_processor = AutoImageProcessor.from_pretrained(
            base_model_path, local_files_only=True
        )
        model = AutoModel.from_pretrained(base_model_path, local_files_only=True)
        vit_classifier = ViTClassifier(model, 4, classifier_type="cosine")

        lora_rank = wandb.config["lora_rank"]
        lora_initialization_strategy = wandb.config["lora_initialization_strategy"]
        num_layer_finetuned = wandb.config["num_layer_finetuned"]

        target_modules = get_target_modules(vit_classifier, num_layer_finetuned)
        if lora_initialization_strategy == "random":
            lora_config = LoraConfig(
                r=lora_rank,
                init_lora_weights=True,
                target_modules=target_modules,
                modules_to_save=["head"],
            )
        elif lora_initialization_strategy == "dora":
            lora_config = LoraConfig(
                r=lora_rank,
                use_dora=True,
                target_modules=target_modules,
                modules_to_save=["head"],
            )
        elif lora_initialization_strategy == "rslora":
            lora_config = LoraConfig(
                r=lora_rank,
                use_rslora=True,
                target_modules=target_modules,
                modules_to_save=["head"],
            )
        peft_vit_classifier = get_peft_model(vit_classifier, lora_config)
        print(peft_vit_classifier.print_trainable_parameters())

        if "print_freq" not in wandb.config:
            wandb.config["print_freq"] = 1
        if "num_workers" not in wandb.config:
            wandb.config["num_workers"] = 2
        if "micro_batch_size" not in wandb.config:
            wandb.config["micro_batch_size"] = wandb.config["batch_size"]
        if "seed" not in wandb.config:
            wandb.config["seed"] = 32
        if "num_epochs" not in wandb.config:
            wandb.config["num_epochs"] = 10

        trainer = Trainer(
            wandb.config,
            peft_vit_classifier,
            image_processor,
            train_dataset_dir,
            test_dataset_dir,
            device,
        )

        trainer.train()

        print("Start Testing")
        results = trainer.test()

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # Log the current best model
            sweep = api.sweep(sweep_id)
            model_output_file = os.path.join(output_dir, wandb.run.name)

            if not sweep.runs:
                trainer.save_model(model_output_file)
            else:
                runs = sorted(
                    sweep.runs,
                    key=lambda run: run.summary.get("test/mean_acc", 0),
                    reverse=True,
                )
                best_run = runs[0]
                best_metric = best_run.summary.get("test/mean_acc", 0)

                if results["mean_acc"] > best_metric:
                    best_run_model_output_file = os.path.join(output_dir, best_run.name)
                    if os.path.exists(best_run_model_output_file):
                        shutil.rmtree(best_run_model_output_file)
                    trainer.save_model(model_output_file)
                elif results["mean_acc"] == best_metric:
                    trainer.save_model(model_output_file)

        for k, v in results.items():
            wandb.run.summary[f"test/{k}"] = v


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to store the best model",
    )
    parser.add_argument(
        "-b", "--base-model-path", type=str, required=True, help="Base model path"
    )
    parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        required=True,
        help="Configuration file for the Wandb sweep",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Wandb project name for the sweep",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=10,
        help="Number of runs for the sweep",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Wandb entity name",
    )
    parser.add_argument(
        "--train-dataset-dir", type=str, help="Train dataset directory", required=True
    )
    parser.add_argument(
        "--test-dataset-dir", type=str, help="Test dataset directory", required=True
    )
    parser.add_argument(
        "--device", default=device, type=str, help="Device to use for sweeping"
    )
    args = vars(parser.parse_args())
    with open(args["configuration"]) as f:
        sweep_config = json.load(f)

    sweep_id = wandb.sweep(
        sweep_config,
        project=args["project_name"],
        entity=args["wandb_entity"],
    )

    def agent_sweep_with_params(sweep_config=None):
        agent_sweep(
            sweep_config,
            sweep_id=sweep_id,
            base_model_path=args["base_model_path"],
            output_dir=args["output_dir"],
            train_dataset_dir=args["train_dataset_dir"],
            test_dataset_dir=args["test_dataset_dir"],
            device=args["device"],
        )

    wandb.agent(
        sweep_id,
        function=agent_sweep_with_params,
        count=args["sweep_count"],
    )
