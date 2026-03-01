from src.config import Config
from transformers import AutoModel, AutoImageProcessor
from src.data.transforms import make_classification_eval_transform
import torchvision.transforms as transforms
from src.model import (
    ViTClassifier,
    MetricPredictor,
    FeatureExtractor,
    create_lora_vit_model,
)
from src.model.utils import get_target_modules
from src.model.resnet101 import (
    ResNet101Classifier,
    ResNetImageProcessor,
    RESNET_DEFAULT_SIZE,
    RESNET_DEFAULT_CROP,
)
from peft import PeftModel
from pathlib import Path
import torch
import os


def get_model(configuration: Config, device: str = "cpu", test_mode: bool = True):
    """
    Load the model based on the configuration

    Returns:
      model: nn.Module
      image_processor: Image transform
      feature_matrix: Optional[torch.Tensor] Used for metric predictor
    """
    if (
        test_mode
        and configuration.model.method == "tuning"
        and configuration.model.model_name != "resnet101"
        and not os.path.exists(configuration.model.peft_adapter_path)
    ):
        raise ValueError(
            f"Adapter path {configuration.model.peft_adapter_path} does not exist."
        )

    if test_mode and configuration.model.method == "no_tuning":
        if not os.path.exists(configuration.model.feature_matrix_path):
            raise ValueError(
                f"Feature matrix path {configuration.model.feature_matrix_path} does not exist."
            )
        else:
            if Path(configuration.model.feature_matrix_path).suffix != ".pt":
                raise ValueError("Feature matrix should be a .pt file.")
    if configuration.model.method == "tuning":
        if configuration.model.model_name == "resnet101":
            # ResNet101 path: no PEFT, full fine-tuning
            pretrained_path = None
            if configuration.model.model_path and os.path.isfile(configuration.model.model_path):
                pretrained_path = configuration.model.model_path
            model = ResNet101Classifier(
                num_classes=configuration.data.num_classes,
                classifier_type=configuration.model.classifier_head,
                pretrained_path=pretrained_path,
                freeze_backbone=getattr(configuration.model, "freeze_backbone", False),
            )
            if test_mode and configuration.model.peft_adapter_path and os.path.isdir(configuration.model.peft_adapter_path):
                ckpt = None
                for name in ("pytorch_model.bin", "model.pt", "model.pth"):
                    p = Path(configuration.model.peft_adapter_path) / name
                    if p.exists():
                        ckpt = p
                        break
                if ckpt is None:
                    for f in Path(configuration.model.peft_adapter_path).iterdir():
                        if f.suffix in (".pt", ".pth", ".bin"):
                            ckpt = f
                            break
                if ckpt and ckpt.exists():
                    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    if isinstance(state, dict):
                        model.load_state_dict(state, strict=False)
            if configuration.inference:
                crop = configuration.inference.crop_size
                resize = getattr(configuration.inference, "resize_size", configuration.inference.resolution)
            else:
                resize, crop = 256, RESNET_DEFAULT_CROP
            image_transforms = make_classification_eval_transform(
                crop_size=crop,
                resize_size=resize,
            )
            resnet_processor = ResNetImageProcessor(resize=resize, crop_size=crop)
            model.to(device)
            return model, image_transforms, resnet_processor

        base_model = AutoModel.from_pretrained(
            configuration.model.model_path, local_files_only=True
        )
        image_transforms = make_classification_eval_transform(
            crop_size=configuration.inference.crop_size,
            resize_size=configuration.inference.resolution * 8 // 7,
        )
        vit_classifier = ViTClassifier(
            base_model,
            num_classes=configuration.data.num_classes,
            classifier_type=configuration.model.classifier_head,
        )
        if test_mode:
            model = PeftModel.from_pretrained(
                vit_classifier, configuration.model.peft_adapter_path
            )
        else:
            model = create_lora_vit_model(
                vit_classifier,
                lora_rank=configuration.peft_training.lora_rank,
                lora_alpha=configuration.peft_training.lora_alpha,
                lora_initialization_strategy=configuration.peft_training.lora_initialization_strategy,
                lora_target_modules=get_target_modules(
                    vit_classifier, configuration.peft_training.num_layer_finetuned
                ),
            )
        model.to(device)
    elif configuration.model.method == "no_tuning":
        base_model = AutoModel.from_pretrained(
            configuration.model.model_path, local_files_only=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            configuration.model.model_path,
            local_files_only=True,
        )
        image_transforms = transforms.Compose(
            [
                transforms.Lambda(
                    lambda images: image_processor(images, return_tensors="pt")[
                        "pixel_values"
                    ].squeeze(0)
                ),
            ]
        )
        if test_mode:
            feature_matrix = torch.load(
                configuration.model.feature_matrix_path, weights_only=False
            )
            model = MetricPredictor(
                feature_matrix=feature_matrix,
                feature_extractor=FeatureExtractor(
                    model=base_model,
                    device=device,
                ),
                device=device,
            )
        else:
            model = base_model

    return model, image_transforms
