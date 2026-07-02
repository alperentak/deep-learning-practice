import sys
import tomllib
from pathlib import Path

from sympy.parsing.sympy_parser import null


class Config:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            sys.exit("config dosyası bulunamadı!")
        self.config_path = config_path
        self.conf = null

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
        if "random_seed" not in conf or conf["random_seed"] == null:
            sys.exit("config: random_seed değeri bulunamadı!")

        if not isinstance(conf["random_seed"], int):
            sys.exit("config: random_seed değeri int olmalıdır!")

        # models
        if "models" not in conf or conf["models"] == null:
            sys.exit("config: models listesi bulunamadı!")
        if not isinstance(conf["models"], list):
            sys.exit("config: models değeri list olmalıdır!")
        model_keys = {
            "name": str,
            "type": str,
            "args": dict,
        }
        for model in conf["models"]:
            for key, value in model_keys.items():
                if key not in model or model[key] == null:
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

        # trainers

        if "trainers" not in conf or conf["trainers"] == null:
            sys.exit("config: trainers listesi bulunamadı!")
        if not isinstance(conf["trainers"], list):
            sys.exit("config: trainers değeri list olmalıdır!")
        trainer_keys = {
            "name": str,
            "optimizer": dict,
            "loss_fn": dict,
            "epochs": int,
            "device": str,
            "save_model": bool,
            "eval_model": bool,
        }
        for trainer in conf["trainers"]:
            for key, value in trainer_keys.items():
                if key not in trainer or trainer[key] == null:
                    sys.exit(f"config: trainer {key} alanı boş bırakılamaz!")
                if not isinstance(trainer[key], value):
                    sys.exit(f"config: trainer {key} değeri {value} olmalıdır!")

            # optimizer
            optimizer = trainer["optimizer"]
            optimizer_types = ["Adam", "SGD"]
            if "type" not in optimizer or optimizer["type"] == null:
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
            if "type" not in loss_fn or loss_fn["type"] == null:
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
