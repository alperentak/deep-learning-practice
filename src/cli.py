import sys
from pathlib import Path

import click

from src.config import Config
from src.pipeline import PipelineRunner
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
def run_pipeline(ctx):
    conf: dict = ctx.obj.get("config").conf
    pipeline_config_path: str = conf["pipeline_config_path"]

    pipeline_runner = PipelineRunner(
        pipeline_config_path=pipeline_config_path,
        config=conf,
    )
    pipeline_runner.run_pipelines()
