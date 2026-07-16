import sys
from pathlib import Path

import click

from src.config import Config
from src.experiment import ExperimentRunner
from src.utils import set_random_seed


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

    set_random_seed(conf.conf["random_seed"])


@cli.command()
@click.pass_context
def run_experiments(ctx):
    """config dosyasında belirtilen experiment'ları çalıştır"""
    conf: dict = ctx.obj.get("config").conf
    experiment_config_path: str = conf["experiment_config_path"]

    experiment_runner = ExperimentRunner(
        experiment_config_path=experiment_config_path,
        config=conf,
    )
    experiment_runner.run_experiments()
