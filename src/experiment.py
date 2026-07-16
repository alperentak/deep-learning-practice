import sys
import tomllib

import click

from src.builders import build_evaluator, build_model, build_trainer
from src.dataset import load_fashion_mnist
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
        }
        unique_experiment_names = list()
        for experiment in config["experiments"]:
            for key, value in experiment_keys.items():
                if key not in experiment or experiment[key] is None:
                    sys.exit(f"experiment config: experiment {key} alanı boş bırakılamaz!")
                if not isinstance(experiment[key], value):
                    sys.exit(
                        f"experiment config: experiment {key} değeri {value} olmalıdır!"
                    )
            if experiment["name"] in unique_experiment_names:
                sys.exit("experiment config: experiment name değeri özgün olmalıdır!")
            unique_experiment_names.append(experiment["name"])

        return config["experiments"]

    def run_experiments(self):
        models_conf = self.config["models"]
        trainers_conf = self.config["trainers"]

        model_save_dir = self.config["model_save_dir"]

        for experiment in self.experiments:
            model_name = experiment["model"]
            trainer_name = experiment["trainer"]

            model_exist = False
            selected_model_conf: dict
            for model_conf in models_conf:
                if model_conf["name"] == model_name:
                    selected_model_conf = model_conf
                    model_exist = True
                    break

            if not model_exist:
                sys.exit(f"{model_name} bulunamadı!")

            trainer_exist = False
            selected_trainer_conf: dict
            for trainer_conf in trainers_conf:
                if trainer_conf["name"] == trainer_name:
                    selected_trainer_conf = trainer_conf
                    trainer_exist = True
                    break

            if not trainer_exist:
                sys.exit(f"{trainer_name} bulunamadı!")

            if self.saver.is_saved(
                model_conf=selected_model_conf,
                trainer_conf=selected_trainer_conf,
            ):
                click.echo()
                click.echo(f"bu model zaten eğitilmiş: {experiment['name']}")
                continue

            click.echo()
            click.secho(f"model eğitiliyor: {experiment['name']}", bold=True)
            model = build_model(selected_model_conf)
            trainer = build_trainer(conf=selected_trainer_conf, model=model)
            evaluator = build_evaluator(selected_trainer_conf)
            train_data_loader, test_data_loader, _ = load_fashion_mnist()

            trainer.train_model(model=model, train_data_loader=train_data_loader)

            self.saver.save_model(
                model=model,
                save_name=experiment["name"],
                model_parameters=selected_model_conf,
                trainer_parameters=selected_trainer_conf,
            )

            evaluator.eval_model(
                model=model,
                test_data_loader=test_data_loader,
                model_name=experiment["name"],
                eval_results_path=self.config["eval_results_path"],
            )
