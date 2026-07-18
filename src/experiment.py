import sys
import tomllib

import click

from src.builders import (build_dataloader, build_dataset, build_evaluator,
                          build_model, build_trainer)
from src.saver import Saver


class ExperimentRunner:
    def __init__(
        self,
        experiment_config_path: str,
        config: dict,
    ):
        self.experiments: dict = self._read_conf(experiment_config_path)
        self.config = config
        self.saver = Saver(save_dir=config["model_save_dir"])

    def _read_conf(self, config_path: str) -> dict:
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            sys.exit(f"experiment config dosyası okunamadı!\n{e}")

        if "experiments" not in config or config["experiments"] is None:
            sys.exit("experiment config: experiments listesi bulunamadı!")
        if not isinstance(config["experiments"], list):
            sys.exit("experiment config: experiments değeri list olmalıdır!")
        experiment_keys = {
            "name": str,
            "model": str,
            "trainer": str,
            "dataset": str,
            "dataloader": str,
        }
        unique_experiment_names = list()
        for experiment in config["experiments"]:
            for key, value in experiment_keys.items():
                if key not in experiment or experiment[key] is None:
                    sys.exit(
                        f"experiment config: experiment {key} alanı boş bırakılamaz!"
                    )
                if not isinstance(experiment[key], value):
                    sys.exit(
                        f"experiment config: experiment {key} değeri {value} olmalıdır!"
                    )
            if experiment["name"] in unique_experiment_names:
                sys.exit("experiment config: experiment name değeri özgün olmalıdır!")
            unique_experiment_names.append(experiment["name"])

        return config["experiments"]

    def run_experiments(self):

        for experiment in self.experiments:
            experiment_conf = {
                "model": dict(),
                "trainer": dict(),
                "dataset": dict(),
                "dataloader": dict(),
            }

            components = [
                {
                    "type": "model",
                    "config_section": "models",
                    "name": experiment["model"],
                },
                {
                    "type": "trainer",
                    "config_section": "trainers",
                    "name": experiment["trainer"],
                },
                {
                    "type": "dataset",
                    "config_section": "datasets",
                    "name": experiment["dataset"],
                },
                {
                    "type": "dataloader",
                    "config_section": "dataloaders",
                    "name": experiment["dataloader"],
                },
            ]

            for component in components:
                component_exist = False
                for comp in self.config[component["config_section"]]:
                    if component["name"] == comp["name"]:
                        experiment_conf[component["type"]] = comp
                        component_exist = True

                if not component_exist:
                    sys.exit(f"{component} bulunamadı!")

            if self.saver.is_saved(
                model_conf=experiment_conf["model"],
                trainer_conf=experiment_conf["trainer"],
            ):
                click.echo()
                click.echo(f"bu model zaten eğitilmiş: {experiment['name']}")
                continue

            click.echo()
            click.secho(f"model eğitiliyor: {experiment['name']}", bold=True)
            model = build_model(experiment_conf["model"])
            trainer = build_trainer(conf=experiment_conf["trainer"], model=model)
            evaluator = build_evaluator(experiment_conf["trainer"])
            train_dataset, test_dataset = build_dataset(experiment_conf["dataset"])
            train_dataloader, test_dataloader = build_dataloader(
                conf=experiment_conf["dataloader"],
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )

            trainer.train_model(model=model, train_data_loader=train_dataloader)

            self.saver.save_model(
                model=model,
                save_name=experiment["name"],
                model_parameters=experiment_conf["model"],
                trainer_parameters=experiment_conf["trainer"],
            )

            evaluator.eval_model(
                model=model,
                test_data_loader=test_dataloader,
                model_name=experiment["name"],
                eval_results_path=self.config["eval_results_path"],
            )
