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
from peft import PeftModel
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_model_efficientnet(checkpoint_path: str, num_classes: int = 4, device: str = "cpu"):
    """
    Load EfficientNet-B0 model and Image Transform
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

    # 1. Initialize EfficientNet-B0 architecture
    model = efficientnet_b0(weights=None)
    
    # Change the last classifier layer to match the 4 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 2. Load weights from best_model.pth file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle the case where the model is trained with DataParallel (has 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # 3. Define Image Transforms
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return model, image_transforms