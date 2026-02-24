import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from src.utils import Evaluator
from src.config import Config
from src.data import KajimaDataset
from src.model import get_model


def eval(configuration: Config, device: str = "cpu", test_mode: bool = True):
    model, image_processor = get_model(configuration, device=device)
    if test_mode:
        dataset = KajimaDataset(
            data_dir=configuration.data.test_dataset_dir,
            transforms=image_processor,
            target_transforms=torch.tensor,
        )
    else:
        dataset = KajimaDataset(
            data_dir=configuration.data.train_dataset_dir,
            transforms=image_processor,
            target_transforms=torch.tensor,
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=configuration.testing.batch_size,
        num_workers=configuration.testing.num_workers,
    )

    # many_idxs = [idx for idx, cls_num in enumerate(dataset.cls_num_list) if cls_num >= 100]
    # few_idxs = [idx for idx, cls_num in enumerate(dataset.cls_num_list) if cls_num <= 20]
    # med_idxs = list(set(range(len(dataset.cls_num_list))) - set(many_idxs) - set(few_idxs))
    # TODO: Remove this hardcoding
    many_idxs = [0]
    few_idxs = [2]
    med_idxs = [1, 3]
    evaluator = Evaluator(config, many_idxs, med_idxs, few_idxs)
    for batch_idx, batch in enumerate(dataloader):
        images = batch[0]
        labels = batch[1]

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        evaluator.update(outputs, labels)

    results = evaluator.evaluate()

    return results


if __name__ == "__main__":
    parser = ArgumentParser()

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        device = "cpu"

    parser.add_argument(
        "-c", "--configuration", type=str, required=True, help="Configuration file"
    )
    parser.add_argument(
        "-d", "--device", type=str, default=device, help="Device to run the model on"
    )

    args = parser.parse_args()

    config = Config.from_json(args.configuration)
    results = eval(config, device=args.device)

    print(results)
