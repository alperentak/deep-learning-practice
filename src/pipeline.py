import sys
import tomllib

import click

from src.builders import build_evaluator, build_model, build_trainer
from src.dataset import load_fashion_mnist
from src.saver import Saver


class PipelineRunner:
    def __init__(
        self,
        pipeline_config_path: str,
        config: dict,
    ):
        self.pipelines: dict = self._read_conf(pipeline_config_path)
        self.config = config
        self.saver = Saver(save_dir=config["model_save_dir"])

    def _read_conf(self, config_path: str) -> dict:
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            sys.exit(f"pipeline config dosyası okunamadı!\n{e}")

        if "pipelines" not in config or config["pipelines"] is None:
            sys.exit("pipeline config: pipelines listesi bulunamadı!")
        if not isinstance(config["pipelines"], list):
            sys.exit("pipeline config: pipelines değeri list olmalıdır!")
        pipeline_keys = {
            "name": str,
            "model": str,
            "trainer": str,
        }
        unique_pipeline_names = list()
        for pipeline in config["pipelines"]:
            for key, value in pipeline_keys.items():
                if key not in pipeline or pipeline[key] is None:
                    sys.exit(f"pipeline config: pipeline {key} alanı boş bırakılamaz!")
                if not isinstance(pipeline[key], value):
                    sys.exit(
                        f"pipeline config: pipeline {key} değeri {value} olmalıdır!"
                    )
            if pipeline["name"] in unique_pipeline_names:
                sys.exit("pipeline config: pipeline name değeri özgün olmalıdır!")
            unique_pipeline_names.append(pipeline["name"])

        return config["pipelines"]

    def run_pipelines(self):
        models_conf = self.config["models"]
        trainers_conf = self.config["trainers"]

        model_save_dir = self.config["model_save_dir"]

        for pipeline in self.pipelines:
            model_name = pipeline["model"]
            trainer_name = pipeline["trainer"]

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
                click.echo(f"bu model zaten eğitilmiş: {pipeline['name']}")
                continue

            click.echo(f"model eğitiliyor: {pipeline['name']}")
            model = build_model(selected_model_conf)
            trainer = build_trainer(conf=selected_trainer_conf, model=model)
            evaluator = build_evaluator(selected_trainer_conf)
            train_data_loader, test_data_loader, _ = load_fashion_mnist()

            trainer.train_model(model=model, train_data_loader=train_data_loader)

            self.saver.save_model(
                model=model,
                save_name=pipeline["name"],
                model_parameters=selected_model_conf,
                trainer_parameters=selected_trainer_conf,
            )

            evaluator.eval_model(
                model=model,
                test_data_loader=test_data_loader,
                model_name=pipeline["name"],
                eval_results_path=self.config["eval_results_path"],
            )
