import sys
from pathlib import Path

import click

from src.config import Config


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
        print("\n".join(f"\t\t{key}: {value}" for key, value in trainer["loss_fn"]["args"].items()))

        print("optimizer:")
        print(f"\ttype: {trainer['optimizer']['type']}")
        print("\targs:")
        print("\n".join(f"\t\t{key}: {value}" for key, value in trainer["optimizer"]["args"].items()))
        print("-" * 30)
    print()
