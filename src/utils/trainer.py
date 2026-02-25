import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BaseImageProcessor
from typing import Optional, Callable, List, Any, TypeVar
from transformers.image_transforms import rescale
from peft import PeftModel
import numpy as np
import time
import datetime
import wandb
import torchvision.transforms as transforms
from src.data import KajimaDataset
from src.data.transforms import (
    RESCALE_FACTOR,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    MaybeToTensor,
)
from src.data.samplers import DownSampler, ClassAwareSampler
from src.utils import (
    Evaluator,
    FocalLoss,
    ClassBalancedLoss,
    LogitAdjustedLoss,
    LADELoss,
    Lion,
    AverageMeter,
)
from src.config import Config

T = TypeVar("T")


def rescale_image(img):
    """
    Rescale the image to a new size.
    """
    return rescale(img, RESCALE_FACTOR)


def stack_image_crop(img_crop):
    """
    Stack the image crop.
    """
    return torch.stack([MaybeToTensor()(img_crop)])


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: PeftModel,
        plain_image_processor: BaseImageProcessor,
        dataset_dir: str,
        test_dataset_dir: str,
        device: str = "cpu",
    ):
        self.device = device
        print("Detected device: ", self.device)

        self.model = model
        self.plain_image_processor = plain_image_processor
        self.cfg = cfg
        self.weight = torch.tensor(cfg.weight).to(self.device)

        self.dataset_dir = dataset_dir
        self.test_dataset_dir = test_dataset_dir

        if self.cfg.micro_batch_size > self.cfg.batch_size:
            self.cfg.micro_batch_size = self.cfg.batch_size

        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(
            self.cfg, self.many_idxs, self.med_idxs, self.few_idxs
        )

    def build_data_loader(self):
        cfg = self.cfg

        resolution = self.plain_image_processor.size["shortest_edge"]
        crop_size = (
            self.plain_image_processor.crop_size["height"],
            self.plain_image_processor.crop_size["width"],
        )
        # Follow train preset of classification task
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(rescale_image),
                MaybeToTensor(),
                transforms.RandomResizedCrop(
                    resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Lambda(rescale_image),
                MaybeToTensor(),
                transforms.Resize(
                    resolution * 8 // 7,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(crop_size),
                transforms.Lambda(stack_image_crop),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

        train_init_dataset = KajimaDataset(
            self.dataset_dir,
            transforms=None,
        )
        train_dataset = KajimaDataset(
            self.dataset_dir,
            transforms=train_transforms,
            target_transforms=torch.tensor,
        )
        test_dataset = KajimaDataset(
            self.test_dataset_dir,
            transforms=test_transforms,
            target_transforms=torch.tensor,
        )

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.many_idxs = [0]
        self.med_idxs = [1, 3]
        self.few_idxs = [2]
        self.init_train_dataloader = self._make_data_loader(
            train_init_dataset,
            sampler="down_sampler",
            batch_size=2,
            num_workers=cfg.num_workers,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
            persistent_workers=False,
            collate_fn=None,
        )

        self.train_dataloader = self._make_data_loader(
            train_dataset,
            sampler=cfg.sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            seed=cfg.seed,
            drop_last=False,
            persistent_workers=True,
            collate_fn=None,
        )

        self.test_dataloader = self._make_data_loader(
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )

        self.accumulate_step = cfg.batch_size // cfg.micro_batch_size

    def _make_data_loader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        sampler: str = None,
        shuffle: bool = True,
        seed: int = 0,
        sampler_size: int = -1,
        sampler_advance: int = 0,
        drop_last: bool = False,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
    ):
        """
        Creates a data loader with the specified parameters.
        """

        if sampler == "class_aware_sampler":
            print("Using class aware sampler")
            sampler = ClassAwareSampler(dataset)
        elif sampler == "down_sampler":
            print("Using down sampler")
            sampler = DownSampler(dataset, n_max=4)
        else:
            sampler = None

        print("using PyTorch data loader")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            # shuffle=shuffle,
        )

        try:
            print(f"# of batches: {len(data_loader):,d}")
        except TypeError:  # data loader has no length
            print("infinite data loader")

        return data_loader

    def build_model(self):
        self.model.to(self.device)

        self.build_optimizer()
        self.build_criterion()

        if self.cfg.init_head == "class_mean":
            self.init_head_class_mean()

    def build_optimizer(self):
        cfg = self.cfg

        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "lion":
            self.optimizer = Lion(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                momentum=getattr(cfg, "momentum", 0.9),
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {cfg.optimizer}")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.num_epochs * len(self.train_dataloader)
        )

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        criterion_type = cfg.criterion_type
        if criterion_type == "focal":
            self.criterion = FocalLoss()
        elif criterion_type == "LA":
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif criterion_type == "CBL":
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif criterion_type == "CE":
            self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        elif criterion_type == "LADE":
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        else:
            raise ValueError(f"Unknown loss {cfg['criterion_type']}")

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class mean")
        with self.model.disable_adapter():
            self.model.eval()

            all_features = []
            all_labels = []
            for batch_idx, batch in enumerate(self.init_train_dataloader):
                images = batch[0]
                labels = batch[1]
                images = images.to(self.device)

                inputs = self.plain_image_processor(images, return_tensors="pt").to(
                    self.device
                )
                outputs = self.model.vit_model(**inputs)

                features = outputs.pooler_output
                all_features.append(features)
                all_labels.append(labels)

            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            sorted_index = all_labels.argsort()
            all_features = all_features[sorted_index]
            all_labels = all_labels[sorted_index]

            unique_labels, counts = torch.unique(all_labels, return_counts=True)

            class_means = [None] * self.num_classes
            idx = 0
            for label, count in zip(unique_labels, counts):
                class_means[label] = all_features[idx : idx + count].mean(
                    dim=0, keepdim=True
                )
                idx += count

            class_means = torch.cat(class_means, dim=0)
            print(class_means.shape)
            class_means = F.normalize(class_means, dim=-1)

            self.model.head.modules_to_save.default.apply_weight(class_means)

    def train(self):
        cfg = self.cfg

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        accuracy_meter = AverageMeter(ema=True)
        cls_accuracy_meter = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        start_time = time.time()
        self.model.train()

        if wandb.run is not None:
            wandb.watch(self.model)

        num_batches = len(self.train_dataloader)
        print(f"Total batches: {num_batches}")

        for epoch in range(cfg.num_epochs):
            end_time = time.time()
            for batch_idx, batch in enumerate(self.train_dataloader):
                data_time.update(time.time() - end_time)

                images = batch[0]
                labels = batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)
                print(labels)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                micro_loss = loss / self.accumulate_step
                micro_loss.backward()

                if ((batch_idx + 1) % self.accumulate_step == 0) or (
                    (batch_idx + 1) == num_batches
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                current_lr = self.optimizer.param_groups[0]["lr"]

                with torch.no_grad():
                    predicted = outputs.argmax(dim=1)
                    correct = predicted.eq(labels).float()
                    accuracy = correct.mean().mul_(100.0)

                loss_meter.update(loss.item())
                accuracy_meter.update(accuracy.item())
                batch_time.update(time.time() - end_time)

                for _c, _y in zip(correct, labels):
                    cls_accuracy_meter[_y.item()].update(_c.mul_(100.0).item(), n=1)
                cls_acc = [cls_accuracy_meter[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_acc))
                many_acc = np.mean(np.array(cls_acc)[self.many_idxs])
                med_acc = np.mean(np.array(cls_acc)[self.med_idxs])
                few_acc = np.mean(np.array(cls_acc)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (cfg.num_epochs - epoch - 1) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch + 1}/{cfg.num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f})"]
                    info += [
                        f"(mean {mean_acc:.4f}) many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"
                    ]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                end_time = time.time()

                n_iter = epoch * num_batches + batch_idx
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/lr": current_lr,
                            "train/loss.val": loss_meter.val,
                            "train/loss.avg": loss_meter.avg,
                            "train/accuracy.val": accuracy_meter.val,
                            "train/accuracy.avg": accuracy_meter.avg,
                            "train/mean_acc": mean_acc,
                            "train/many_acc": many_acc,
                            "train/med_acc": med_acc,
                            "train/few_acc": few_acc,
                        },
                        step=n_iter,
                    )

            self.scheduler.step()
            torch.cuda.empty_cache()
        print("Finish training")
        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")

    @torch.no_grad()
    def test(self, mode="test"):
        self.model.eval()
        if mode == "train":
            dataloader = self.train_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        for batch_idx, batch in enumerate(dataloader):
            images = batch[0]
            labels = batch[1]
            images = images.to(self.device)
            labels = labels.to(self.device)

            bz, crop_num, c, h, w = images.size()
            images = images.view(bz * crop_num, c, h, w)
            outputs = self.model(images)

            self.evaluator.update(outputs, labels)

        results = self.evaluator.evaluate()

        return results

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.plain_image_processor.save_pretrained(output_dir)

    def load_model(self, model_dir):
        self.model.from_pretrained(model_dir)
        self.plain_image_processor.from_pretrained(model_dir)
