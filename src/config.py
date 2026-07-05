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

        # random_seed
        if "random_seed" not in conf or conf["random_seed"] is None:
            sys.exit("config: random_seed değeri bulunamadı!")

        if not isinstance(conf["random_seed"], int):
            sys.exit("config: random_seed değeri int olmalıdır!")

        # model_save_dir
        if "model_save_dir" not in conf or conf["model_save_dir"] is None:
            sys.exit("config: model_save_dir değeri bulunamadı!")

        if not isinstance(conf["model_save_dir"], str):
            sys.exit("config: model_save_dir değeri int olmalıdır!")

        # eval_results_path
        if "eval_results_path" not in conf or conf["eval_results_path"] is None:
            sys.exit("config: eval_results_path değeri bulunamadı!")

        if not isinstance(conf["eval_results_path"], str):
            sys.exit("config: eval_results_path değeri int olmalıdır!")

        # models

        if "models" not in conf or conf["models"] is None:
            sys.exit("config: models listesi bulunamadı!")
        if not isinstance(conf["models"], list):
            sys.exit("config: models değeri list olmalıdır!")
        model_keys = {
            "name": str,
            "type": str,
            "args": dict,
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
