import sys
from pathlib import Path

import click
import torch

from src.builders import build_model, build_trainer
from src.config import Config
from src.dataset import load_fashion_mnist
from src.saver import save_model
from src.trainer import Trainer


@click.group()
@click.option(
    "-c", "--config", default="./config.toml", help="config dosyasının dosya yolu"
)
@click.help_option("-h", "--help", help="bu yardım mesajını gösterir")
@click.pass_context
def cli(ctx: click.Context, config: str):
    """
    deep learning modelleri ve veri setleri ile çalışmak için bir cli aracı
    """
    if not Path(config).exists():
        sys.exit(f"config dosyası bulunamadı! {config}")
    conf = Config(config)
    conf.read_config()

    ctx.ensure_object(dict)
    ctx.obj["config"] = conf


@cli.command()
@click.pass_context
def list_models(ctx):
    """
    model yapılandırmalarını listeler
    """
    print()
    for model in ctx.obj.get("config").conf["models"]:
        print("-" * 30)
        print(f"name: {model['name']}")
        print(f"type: {model['type']}")
        print("args:")
        print("\n".join(f"\t{key}: {value}" for key, value in model["args"].items()))
        print("-" * 30)
    print()


@cli.command()
@click.pass_context
def list_trainers(ctx):
    """
    trainer yapılandırmalarını listeler
    """
    print()
    for trainer in ctx.obj.get("config").conf["trainers"]:
        print("-" * 30)
        print(f"name: {trainer['name']}")
        print(f"epochs: {trainer['epochs']}")
        print(f"device: {trainer['device']}")
        print(f"save_model: {trainer['save_model']}")
        print(f"eval_model: {trainer['eval_model']}")

        print("loss_fn:")
        print(f"\ttype: {trainer['loss_fn']['type']}")
        print("\targs:")
        print(
            "\n".join(
                f"\t\t{key}: {value}"
                for key, value in trainer["loss_fn"]["args"].items()
            )
        )

        print("optimizer:")
        print(f"\ttype: {trainer['optimizer']['type']}")
        print("\targs:")
        print(
            "\n".join(
                f"\t\t{key}: {value}"
                for key, value in trainer["optimizer"]["args"].items()
            )
        )
        print("-" * 30)
    print()


@cli.command()
@click.option("-m", "--model-name", required=True, help="eğitilecek model ismi")
@click.option("-t", "--trainer-name", required=True, help="kullanılacak trainer ismi")
@click.option("-n", "--saved-name", required=True, help="kaydedilen modelin ismi")
@click.pass_context
def train_model(ctx, model_name: str, trainer_name: str, saved_name: str):
    """
    trainer ile modeli eğitir ve kaydeder
    """
    conf: dict = ctx.obj.get("config").conf

    models_conf = conf["models"]
    trainers_conf = conf["trainers"]

    model_exist = False
    selected_model: dict
    for model in models_conf:
        if model["name"] == model_name:
            selected_model = model
            model_exist = True
            break

    if not model_exist:
        sys.exit(f"{model_name} bulunamadı!")

    trainer_exist = False
    selected_trainer: dict
    for trainer in trainers_conf:
        if trainer["name"] == trainer_name:
            selected_trainer = trainer
            trainer_exist = True
            break

    if not trainer_exist:
        sys.exit(f"{trainer_name} bulunamadı!")

    model: torch.nn.Module = build_model(conf=selected_model)
    trainer: Trainer = build_trainer(conf=selected_trainer, model=model)

    train_data_loader, _, _ = load_fashion_mnist()

    trainer.train_model(model=model, train_data_loader=train_data_loader)

    save_model(
        model=model,
        save_dir=conf["model_save_dir"],
        name=saved_name,
        model_parameters=selected_model,
        trainer_parameters=selected_trainer,
    )
