import os
from tqdm.auto import tqdm
import torch
from pathlib import Path
from argparse import ArgumentParser
from src.config import Config
from src.utils import Trainer
from transformers import AutoImageProcessor
from src.model import get_model, FeatureExtractor
from src.data import KajimaDataset
from torch.utils.data import DataLoader
from collections import defaultdict


def train(
    configuration: Config,
    device: str = "cuda:0",
    output_dir: str = None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model, processor = get_model(configuration, device=device, test_mode=False)
    method = configuration.model.method
    if method == "tuning":
        trainer = Trainer(
            cfg=configuration.peft_training,
            model=model,
            plain_image_processor=AutoImageProcessor.from_pretrained(
                configuration.model.model_path
            ),
            device=device,
            dataset_dir=configuration.data.train_dataset_dir,
            test_dataset_dir=configuration.data.test_dataset_dir,
        )
        trainer.train()
        trainer.test()

        if output_dir is not None:
            trainer.save_model(output_dir)
    elif method == "no_tuning":
        feature_extractor = FeatureExtractor(model=model, device=device)
        dataset = KajimaDataset(
            data_dir=configuration.data.train_dataset_dir,
            transforms=processor,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=configuration.metric_training.batch_size,
            num_workers=configuration.metric_training.num_workers,
        )
        results = defaultdict(list)
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            outputs = feature_extractor.extract_feature(images)

            for output, label in zip(outputs.to("cpu"), labels):
                results[label.item()].append(output)

        if output_dir is not None:
            torch.save(results, str(Path(output_dir) / "feature_matrix.pt"))

    return True


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--configuration", type=str, required=True, help="Configuration file"
    )
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("-o", "--output-dir", type=str, default=None)

    args = parser.parse_args()
    config = Config.from_json(args.configuration)
    train(
        configuration=config,
        device=args.device,
        output_dir=args.output_dir,
    )
