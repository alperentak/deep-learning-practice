import random
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset import ImageDataset
from src.evaluator import Evaluator
from src.models import BaselineModel, CNNModel
from src.trainer import Trainer


def build_model(conf: dict) -> torch.nn.Module:
    """
    config değerleriyle model oluşturur
    """
    try:
        if conf["type"] == "BaselineModel":
            model: torch.nn.Module = BaselineModel(**conf["kwargs"])
        elif conf["type"] == "CNNModel":
            model: torch.nn.Module = CNNModel(**conf["kwargs"])
        else:
            sys.exit("model type bulunamadı!")
    except Exception as e:
        sys.exit(f"model oluşturulamadı!\n{e}")

    return model


def build_trainer(conf: dict, model: torch.nn.Module) -> Trainer:
    """
    config değerleriyle trainer oluşturur
    """
    try:
        if conf["loss_fn"]["type"] == "L1Loss":
            loss_fn = torch.nn.L1Loss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "MSELoss":
            loss_fn = torch.nn.MSELoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "BCELoss":
            loss_fn = torch.nn.BCELoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "BCEWithLogitsLoss":
            loss_fn = torch.nn.BCEWithLogitsLoss(**conf["loss_fn"]["kwargs"])
        else:
            sys.exit("loss type bulunamadı!")
    except Exception as e:
        sys.exit(f"loss fonksiyonu oluşturulamadı!\n{e}")

    try:
        if conf["optimizer"]["type"] == "Adam":
            optimizer = torch.optim.Adam(
                params=model.parameters(), **conf["optimizer"]["kwargs"]
            )
        elif conf["optimizer"]["type"] == "SGD":
            optimizer = torch.optim.SGD(
                params=model.parameters(), **conf["optimizer"]["kwargs"]
            )
        else:
            sys.exit("optimizer type bulunamadı!")
    except Exception as e:
        sys.exit(f"optimizer oluşturulamadı!\n{e}")

    trainer: Trainer = Trainer(
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=conf["epochs"],
        device=conf["device"],
    )

    return trainer


def build_evaluator(conf: dict) -> Evaluator:
    """
    config değerleriyle evaluator oluşturur
    """
    try:
        if conf["loss_fn"]["type"] == "L1Loss":
            loss_fn = torch.nn.L1Loss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "MSELoss":
            loss_fn = torch.nn.MSELoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "BCELoss":
            loss_fn = torch.nn.BCELoss(**conf["loss_fn"]["kwargs"])
        elif conf["loss_fn"]["type"] == "BCEWithLogitsLoss":
            loss_fn = torch.nn.BCEWithLogitsLoss(**conf["loss_fn"]["kwargs"])
        else:
            sys.exit("loss type bulunamadı!")
    except Exception as e:
        sys.exit(f"loss fonksiyonu oluşturulamadı!\n{e}")

    evaluator: Evaluator = Evaluator(
        loss_fn=loss_fn,
        device=conf["device"],
    )

    return evaluator


def build_dataset(conf: dict) -> tuple[Dataset, Dataset]:
    transforms_map = {
        "Resize": transforms.Resize,
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
        "ToTensor": transforms.ToTensor,
    }

    transforms_dict = {
        "train": list(),
        "test": list(),
    }
    transform = {
        "train": None,
        "test": None,
    }
    for split in {"train", "test"}:
        transforms_conf = conf[split]["transforms"]
        for t in transforms_conf:
            transform_class = transforms_map.get(t["name"])
            if transform_class is None:
                click.echo(f"datasets: {t['name']} bulunamadı!")
                continue
            transform_args = t.get("args", [])
            try:
                transforms_dict[split].append(transform_class(*transform_args))
            except Exception as e:
                sys.exit(f"transform {transform_class} eklenemedi!\n{e}")

        if len(transforms_dict[split]) > 0:
            transform[split] = transforms.Compose(transforms_dict[split])
        else:
            transform[split] = None

    data_dir = Path(conf["data_dir"])
    class_names = sorted(entry.name for entry in data_dir.iterdir() if entry.is_dir())

    if not class_names:
        sys.exit(f"{data_dir}: class klasörleri bulunamadı!")

    classes = {class_name: i for i, class_name in enumerate(class_names)}

    train_list = list()
    test_list = list()

    image_extensions = {".png", ".jpg", ".jpeg"}
    train_test_split = conf["train_test_split"]
    for class_name, index in classes.items():
        class_dir = data_dir / class_name
        image_paths = [
            Path(image_path)
            for image_path in class_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in image_extensions
        ]

        random.shuffle(image_paths)

        split_index = int(len(image_paths) * train_test_split)

        for img_path in image_paths[:split_index]:
            train_list.append({"path": img_path, "label": index})
        for img_path in image_paths[split_index:]:
            test_list.append({"path": img_path, "label": index})

    train_dataset = ImageDataset(
        classes=classes, data_list=train_list, transform=transform["train"]
    )
    test_dataset = ImageDataset(
        classes=classes, data_list=test_list, transform=transform["test"]
    )

    return train_dataset, test_dataset
