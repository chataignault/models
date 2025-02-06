from datasets import load_dataset

# from torchtune import ConcatDataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    RandomHorizontalFlip,
    Pad,
)
from functools import partial
import torch
from torch.utils.data import DataLoader


def get_transforms(
    examples, device: str, channels_last: bool = True, zero_pad_images: bool = False
):
    t_ = [
        RandomHorizontalFlip(),
        ToTensor(),
    ]
    if zero_pad_images:
        t_.append(Pad(2))
    t_.append(Lambda(lambda t: (t * 2) - 1))
    transform = Compose(transforms=t_)

    if channels_last:
        examples["pixel_values"] = [
            transform(image.convert("L")).to(device).permute(1, 2, 0)
            for image in examples["image"]
        ]
    else:
        examples["pixel_values"] = [
            transform(image.convert("L")).to(device) for image in examples["image"]
        ]

    del examples["image"]
    return examples


def get_dataloader(
    batch_size: int,
    device: str,
    channels_last: bool = True,
    zero_pad_images: bool = False,
):
    dataset = load_dataset("fashion_mnist")

    transforms_dev = partial(
        get_transforms,
        device=device,
        channels_last=channels_last,
        zero_pad_images=zero_pad_images,
    )
    transformed_dataset = dataset.with_transform(transforms_dev).remove_columns("label")

    dataloader = DataLoader(
        torch.utils.data.ConcatDataset(
            [transformed_dataset["train"], transformed_dataset["test"]]
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device=device),
    )
    return dataloader
