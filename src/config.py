import os
import json
from typing import Literal, Annotated
from pydantic import BaseModel, AfterValidator


def check_file_exists(file_path: str):
    if not os.path.exists(file_path):
        print(f"Warning: Model file {file_path} does not exist.")
    return file_path


class ModelConfig(BaseModel):
    method: Literal["tuning", "no_tuning"]
    model_name: str
    model_path: Annotated[str, AfterValidator(check_file_exists)]
    feature_matrix_path: str | None = None
    peft_adapter_path: str | None = None
    classifier_head: Literal["cosine", "linear", "layernorm", "l2norm"] = "cosine"


class PeftTrainingConfig(BaseModel):
    batch_size: int = 8
    criterion_type: Literal["CBL", "LA", "focal", "CE", "LADE"]
    weight: list[float] = [1.0] * 4
    init_head: Literal["class_mean", "no"]
    lora_initialization_strategy: Literal["dora", "random", "rslora"]
    lora_rank: int = 16
    lr: float = 1e-6
    micro_batch_size: int = 8
    num_epochs: int = 10
    num_layer_finetuned: int | None = None  # None means all layers are finetuned
    num_workers: int = 0
    optimizer: Literal["adam", "lion", "adamw"] = "adam"
    print_freq: int = 1
    sampler: Literal["down_sampler", "class_aware_sampler", "random_sampler", ""] = "class_aware_sampler"
    seed: int = 32
    weight_decay: float = 0.0


class DataConfig(BaseModel):
    train_dataset_dir: Annotated[str, AfterValidator(check_file_exists)]
    test_dataset_dir: Annotated[str, AfterValidator(check_file_exists)]
    num_classes: int
    val_dataset_dir: str | None = None


class MetricTrainingConfig(BaseModel):
    batch_size: int = 8
    num_workers: int = 0


class InferenceConfig(BaseModel):
    resolution: int
    crop_size: int


class TestingConfig(BaseModel):
    batch_size: int = 2
    num_workers: int = 0


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False


class Config(BaseModel):
    model: ModelConfig
    peft_training: PeftTrainingConfig | None = None
    metric_training: MetricTrainingConfig | None = None
    testing: TestingConfig | None = None
    inference: InferenceConfig | None = None
    data: DataConfig
    app: AppConfig | None = None

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, "r") as f:
            config_dict = json.load(f)

        # Model config
        model_dict = config_dict.get("model", {})

        # Training config based on model method
        training_dict = config_dict.get("training", {})
        if model_dict.get("method") == "tuning":
            config_dict["peft_training"] = training_dict
            config_dict["metric_training"] = None
        elif model_dict.get("method") == "no_tuning":
            config_dict["metric_training"] = training_dict
            config_dict["peft_training"] = None
        else:
            raise ValueError("Invalid model method. Choose 'tuning' or 'no_tuning'.")

        # Remove the original training key
        if "training" in config_dict:
            del config_dict["training"]

        # Parse the config with Pydantic
        print(config_dict)
        return cls.model_validate(config_dict)

    def to_json(self, json_file: str):
        """Save the configuration to a JSON file"""
        # Convert to dict, combining training configs based on method
        config_dict = self.model_dump(exclude_none=True)

        # Extract the appropriate training config based on method
        if self.model.method == "tuning" and self.peft_training:
            config_dict["training"] = config_dict.pop("peft_training")
        elif self.model.method == "no_tuning" and self.metric_training:
            config_dict["training"] = config_dict.pop("metric_training")

        # Remove the unused training config
        if "peft_training" in config_dict:
            del config_dict["peft_training"]
        if "metric_training" in config_dict:
            del config_dict["metric_training"]

        # Write to file
        with open(json_file, "w") as f:
            json.dump(config_dict, f, indent=2)
