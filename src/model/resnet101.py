"""
ResNet101 backbone and classifier for replacement of DINOv2.
"""
import torch
import torch.nn as nn
from .classifiers import (
    LinearClassifierHead,
    CosineClassifier,
    LayerNormedClassifier,
    L2NormedClassifier,
)

RESNET101_FEATURE_DIM = 2048  # 512 * Bottleneck.expansion


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

    def forward_features(self, x):
        """Return pooled feature vector (no classification head)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def resnet101_backbone():
    """ResNet101 without the final FC layer (for use as backbone)."""
    net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1000)
    net.fc = nn.Identity()
    return net


def resnet101(num_classes=1000):
    """Full ResNet101 with classification head (e.g. for linear head only)."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


class ResNet101Classifier(nn.Module):
    """ResNet101 backbone + configurable classifier head (cosine, linear, etc.)."""

    def __init__(
        self,
        num_classes: int,
        classifier_type: str = "linear",
        pretrained_path: str | None = None,
        freeze_backbone: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.backbone = resnet101_backbone()
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            self.backbone.load_state_dict(state, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if classifier_type == "linear":
            self.head = LinearClassifierHead(RESNET101_FEATURE_DIM, num_classes, dtype)
        elif classifier_type == "cosine":
            self.head = CosineClassifier(RESNET101_FEATURE_DIM, num_classes, dtype)
        elif classifier_type == "layernorm":
            self.head = LayerNormedClassifier(RESNET101_FEATURE_DIM, num_classes, dtype)
        elif classifier_type == "l2norm":
            self.head = L2NormedClassifier(RESNET101_FEATURE_DIM, num_classes, dtype)
        else:
            raise NotImplementedError(f"classifier_type={classifier_type}")

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        features = self.backbone.forward_features(x)
        return self.head(features)


# Default input size for ResNet (ImageNet-style)
RESNET_DEFAULT_SIZE = 224
RESNET_DEFAULT_CROP = 224


class ResNetImageProcessor:
    """
    Fake 'image processor' so Trainer can use the same data pipeline.
    Has .size, .crop_size and __call__(images, return_tensors="pt") -> object with .pixel_values and .to(device).
    """

    def __init__(self, resize=256, crop_size=224):
        self.size = {"shortest_edge": crop_size}
        self.crop_size = {"height": crop_size, "width": crop_size}
        self._resize = resize
        self._crop_size = crop_size

    def __call__(self, images, return_tensors="pt"):
        import torchvision.transforms as T
        from src.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        def _to_tensor(img):
            if isinstance(img, torch.Tensor):
                x = img.float()
                if x.max() > 1.0:
                    x = x / 255.0
                t_tensor = T.Compose([
                    T.Resize(self._resize, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(self._crop_size),
                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ])
                return t_tensor(x)
            from src.data.transforms import MaybeToTensor
            from transformers.image_transforms import rescale
            RESCALE = 0.00392156862745098
            t_pil = T.Compose([
                T.Lambda(lambda i: rescale(i, RESCALE)),
                MaybeToTensor(),
                T.Resize(self._resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self._crop_size),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
            return t_pil(img)

        # Batch tensor (B, C, H, W): process as single batch, do not wrap in list (would create wrong 5D shape)
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            pixel_values = _to_tensor(images)
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            tensors = [_to_tensor(img) for img in images]
            pixel_values = torch.stack(tensors)

        class _ProcessorOutput:
            def __init__(self, pv):
                self.pixel_values = pv
            def to(self, device):
                o = _ProcessorOutput(self.pixel_values.to(device))
                return o

        return _ProcessorOutput(pixel_values)

    def save_pretrained(self, path):
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "resnet_processor_config.json", "w") as f:
            json.dump({"resize": self._resize, "crop_size": self._crop_size}, f)

    @classmethod
    def from_pretrained(cls, path):
        import json
        from pathlib import Path
        cfg_path = Path(path) / "resnet_processor_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cls(resize=cfg.get("resize", 256), crop_size=cfg.get("crop_size", 224))
        return cls()
