import sys

import torch

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
