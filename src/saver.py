import json
import sys
from pathlib import Path

import torch

from src.builders import build_model


class Saver:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_info_path = self.save_dir / "saved_models.json"

    def save_model(
        self,
        model: torch.nn.Module,
        save_name: str,
        model_parameters: dict,
        trainer_parameters: dict,
    ):
        """
        model state_dict kaydeder ve kayıt bilgilerini tutar
        """

        model_save_path = self.save_dir / f"{save_name}.pt"

        try:
            torch.save(obj=model.state_dict(), f=model_save_path)
        except Exception as e:
            sys.exit(f"model kaydedilemedi!\n{e}")

        save_info = {
            "path": str(model_save_path),
            "model": model_parameters,
            "trainer": trainer_parameters,
        }

        if not self.save_info_path.exists():
            self.save_info_path.write_text("{}")

        all_saved_models = dict()
        try:
            with open(self.save_info_path, "r", encoding="utf-8") as f:
                all_saved_models = json.load(f)
        except Exception as e:
            sys.exit(f"kayıt dosyası okunamadı!{e}")

        all_saved_models[save_name] = save_info

        try:
            with open(self.save_info_path, "w", encoding="utf-8") as f:
                json.dump(all_saved_models, f, indent=4, ensure_ascii=False)
        except Exception as e:
            sys.exit(f"kayıt dosyasına yazılamadı!\n{e}")

    def load_model(self, save_name: str) -> torch.nn.Module:
        if not self.save_info_path.exists():
            sys.exit("kayıt içeriği bulunamadı!")

        try:
            with open(self.save_info_path, "r", encoding="utf-8") as f:
                loaded_info = json.load(f)
        except Exception as e:
            sys.exit(f"kayıt içeriği okunamadı!\n{e}")

        if save_name not in loaded_info.keys():
            sys.exit("bu isimde kayıtlı model bulunamadı")

        model_info = loaded_info[save_name]["model"]
        model_save_dir = loaded_info[save_name]["path"]

        model: torch.nn.Module = build_model(model_info)

        model.load_state_dict(torch.load(f=model_save_dir))

        return model

    def is_saved(self, model_conf: dict, trainer_conf) -> bool:
        """
        aynı model ve trainer parametrelerine sahip bir modelin kaydedilme durumunu kontrol eder
        """
        if not self.save_info_path.exists():
            return False

        try:
            with open(self.save_info_path, "r", encoding="utf-8") as f:
                saved_models = json.load(f)
        except Exception as e:
            sys.exit(f"kayıt içeriği okunamadı!\n{e}")

        for _, saved_conf in saved_models.items():
            saved_model_conf = saved_conf["model"]
            saved_trainer_conf = saved_conf["trainer"]

            if saved_model_conf == model_conf and saved_trainer_conf == trainer_conf:
                return True
        return False
