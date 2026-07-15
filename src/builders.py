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
            model: torch.nn.Module = BaselineModel(**conf["args"])
        elif conf["type"] == "CNNModel":
            model: torch.nn.Module = CNNModel(**conf["args"])
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
            loss_fn = torch.nn.L1Loss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "MSELoss":
            loss_fn = torch.nn.MSELoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "BCELoss":
            loss_fn = torch.nn.BCELoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "BCEWithLogitsLoss":
            loss_fn = torch.nn.BCEWithLogitsLoss(**conf["loss_fn"]["args"])
        else:
            sys.exit("loss type bulunamadı!")
    except Exception as e:
        sys.exit(f"loss fonksiyonu oluşturulamadı!\n{e}")

    try:
        if conf["optimizer"]["type"] == "Adam":
            optimizer = torch.optim.Adam(
                params=model.parameters(), **conf["optimizer"]["args"]
            )
        elif conf["optimizer"]["type"] == "SGD":
            optimizer = torch.optim.SGD(
                params=model.parameters(), **conf["optimizer"]["args"]
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
            loss_fn = torch.nn.L1Loss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "MSELoss":
            loss_fn = torch.nn.MSELoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "CrossEntropyLoss":
            loss_fn = torch.nn.CrossEntropyLoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "BCELoss":
            loss_fn = torch.nn.BCELoss(**conf["loss_fn"]["args"])
        elif conf["loss_fn"]["type"] == "BCEWithLogitsLoss":
            loss_fn = torch.nn.BCEWithLogitsLoss(**conf["loss_fn"]["args"])
        else:
            sys.exit("loss type bulunamadı!")
    except Exception as e:
        sys.exit(f"loss fonksiyonu oluşturulamadı!\n{e}")

    evaluator: Evaluator = Evaluator(
        loss_fn=loss_fn,
        device=conf["device"],
    )

    return evaluator


def build_dataset(conf: dict) -> Dataset:
    transforms_conf: list = conf["transforms"]
    transforms_map = {
        "Resize": transforms.Resize,
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
        "ToTensor": transforms.ToTensor,
    }

    transforms_list = list()
    for t in transforms_conf:
        transform_class = transforms_map.get(t["name"])
        if transform_class is None:
            click.echo(f"datasets: {t['name']} bulunamadı!")
            continue
        transform_args = t.get("args", [])
        try:
            transforms_list.append(transform_class(*transform_args))
        except Exception as e:
            sys.exit(f"transform {transform_class} eklenemedi!\n{e}")

    if len(transforms_list) > 0:
        transform = transforms.Compose(transforms_list)
    else:
        transform = None

    data_dir = Path(conf["data_dir"])
    class_names = sorted(entry.name for entry in data_dir.iterdir() if entry.is_dir())

    if not class_names:
        sys.exit(f"{data_dir}: class klasörleri bulunamadı!")

    classes = {class_name: i for i, class_name in enumerate(class_names)}

    data_list = list()
    for class_name, index in classes.items():
        class_dir = data_dir / class_name
        image_paths = [Path(image_path) for image_path in class_dir.iterdir()]

        image_extensions = {".png", ".jpg", ".jpeg"}
        for img_path in image_paths:
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                data_list.append({"path": img_path, "label": index})

    image_dataset = ImageDataset(
        classes=classes, data_list=data_list, transform=transform
    )

    return image_dataset
