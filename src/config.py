import sys
import tomllib
from pathlib import Path


class Config:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            sys.exit("config dosyası bulunamadı!")
        self.config_path = config_path

    def read_config(self):
        """
        config dosyasını kontrol eder ve conf içerisine okur
        """
        with open(file=self.config_path, mode="rb") as f:
            try:
                conf = tomllib.load(f)
            except Exception as e:
                sys.exit(f"config okunamadı:\n{e}")

        config_keys = {
            "random_seed": int,
            "experiment_config_path": str,
            "model_save_dir": str,
            "eval_results_path": str,
            "datasets": list,
            "models": list,
            "trainers": list,
            "evaluators": list,
        }
        for key, data_type in config_keys.items():
            if key not in conf or conf[key] is None:
                sys.exit(f"config: model {key} alanı boş bırakılamaz!")
            if not isinstance(conf[key], data_type):
                sys.exit(f"config: model {key} değeri {data_type} türünde olmalıdır!")

        # datasets

        if "datasets" not in conf or conf["datasets"] is None:
            sys.exit("config: datasets listesi bulunamadı!")
        if not isinstance(conf["datasets"], list):
            sys.exit("config: datasets değeri list olmalıdır!")
        dataset_keys = {
            "name": str,
            "data_dir": str,
            "train": dict,
            "test": dict,
        }
        unique_dataset_names = list()
        for dataset in conf["datasets"]:
            for key, value in dataset_keys.items():
                if key not in dataset or dataset[key] is None:
                    sys.exit(f"config: dataset {key} alanı boş bırakılamaz!")
                if not isinstance(dataset[key], value):
                    sys.exit(f"config: dataset {key} değeri {value} olmalıdır!")
            if dataset["name"] in unique_dataset_names:
                sys.exit("config: dataset name değeri özgün olmalıdır!")
            unique_dataset_names.append(dataset["name"])

        # models

        if "models" not in conf or conf["models"] is None:
            sys.exit("config: models listesi bulunamadı!")
        if not isinstance(conf["models"], list):
            sys.exit("config: models değeri list olmalıdır!")
        model_keys = {
            "name": str,
            "type": str,
            "kwargs": dict,
        }
        unique_model_names = list()
        for model in conf["models"]:
            for key, value in model_keys.items():
                if key not in model or model[key] is None:
                    sys.exit(f"config: model {key} alanı boş bırakılamaz!")
                if not isinstance(model[key], value):
                    sys.exit(f"config: model {key} değeri {value} olmalıdır!")
            model_types = ["BaselineModel", "CNNModel"]
            if model["type"] not in model_types:
                sys.exit(
                    "config: model type tanınamadı: "
                    + f"{model['type']}"
                    + "\n"
                    + "tanınan modeller:"
                    + "\n"
                    + "\n".join(f"\t- {i}" for i in model_types)
                )
            if model["name"] in unique_model_names:
                sys.exit("config: model name değeri özgün olmalıdır!")
            unique_model_names.append(model["name"])

        # trainers

        if "trainers" not in conf or conf["trainers"] is None:
            sys.exit("config: trainers listesi bulunamadı!")
        if not isinstance(conf["trainers"], list):
            sys.exit("config: trainers değeri list olmalıdır!")
        trainer_keys = {
            "name": str,
            "optimizer": dict,
            "loss_fn": dict,
            "epochs": int,
            "device": str,
        }
        unique_trainer_names = list()
        for trainer in conf["trainers"]:
            for key, value in trainer_keys.items():
                if key not in trainer or trainer[key] is None:
                    sys.exit(f"config: trainer {key} alanı boş bırakılamaz!")
                if not isinstance(trainer[key], value):
                    sys.exit(f"config: trainer {key} değeri {value} olmalıdır!")

            if trainer["name"] in unique_trainer_names:
                sys.exit("config: trainer name değeri özgün olmalıdır!")
            unique_trainer_names.append(trainer["name"])

            # optimizer
            optimizer = trainer["optimizer"]
            optimizer_types = ["Adam", "SGD"]
            if "type" not in optimizer or optimizer["type"] is None:
                sys.exit("config: optimizer type değeri belirtilmelidir!")
            if optimizer["type"] not in optimizer_types:
                sys.exit(
                    f"config: optimizer type tanınamadı: {optimizer['type']}"
                    + "\n"
                    + "tanınan type değerleri:"
                    + "\n"
                    + "\n".join(f"\t- {i}" for i in optimizer_types)
                )

            # loss_fn
            loss_fn = trainer["loss_fn"]
            loss_fn_types = [
                "L1Loss",
                "MSELoss",
                "CrossEntropyLoss",
                "BCELoss",
                "BCEWithLogitsLoss",
            ]
            if "type" not in loss_fn or loss_fn["type"] is None:
                sys.exit("config: loss_fn type değeri belirtilmelidir!")
            if loss_fn["type"] not in loss_fn_types:
                sys.exit(
                    f"config: loss_fn type tanınamadı: {loss_fn['type']}"
                    + "\n"
                    + "tanınan type değerleri:"
                    + "\n"
                    + "\n".join(f"\t- {i}" for i in loss_fn_types)
                )

        # evaluators

        if "evaluators" not in conf or conf["evaluators"] is None:
            sys.exit("config: evaluators listesi bulunamadı!")
        if not isinstance(conf["evaluators"], list):
            sys.exit("config: evaluators değeri list olmalıdır!")
        evaluator_keys = {
            "name": str,
            "loss_fn": dict,
            "device": str,
        }
        unique_evaluator_names = list()
        for evaluator in conf["evaluators"]:
            for key, value in evaluator_keys.items():
                if key not in evaluator or evaluator[key] is None:
                    sys.exit(f"config: evaluator {key} alanı boş bırakılamaz!")
                if not isinstance(evaluator[key], value):
                    sys.exit(f"config: evaluator {key} değeri {value} olmalıdır!")

            if evaluator["name"] in unique_evaluator_names:
                sys.exit("config: evaluator name değeri özgün olmalıdır!")
            unique_evaluator_names.append(evaluator["name"])

            # loss_fn
            loss_fn = evaluator["loss_fn"]
            loss_fn_types = [
                "L1Loss",
                "MSELoss",
                "CrossEntropyLoss",
                "BCELoss",
                "BCEWithLogitsLoss",
            ]
            if "type" not in loss_fn or loss_fn["type"] is None:
                sys.exit("config: loss_fn type değeri belirtilmelidir!")
            if loss_fn["type"] not in loss_fn_types:
                sys.exit(
                    f"config: loss_fn type tanınamadı: {loss_fn['type']}"
                    + "\n"
                    + "tanınan type değerleri:"
                    + "\n"
                    + "\n".join(f"\t- {i}" for i in loss_fn_types)
                )

        self.conf: dict = conf
