import json
import sys
from pathlib import Path

import torch

from src.builders import build_model


def save_model(
    model: torch.nn.Module,
    save_dir: str,
    name: str,
    model_parameters: dict,
    trainer_parameters: dict,
):
    """
    model state_dict kaydeder ve kayıt bilgilerini tutar
    """
    model_save_dir = Path(save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = model_save_dir / f"{name}.pt"

    try:
        torch.save(obj=model.state_dict(), f=model_save_path)
    except Exception as e:
        sys.exit(f"model kaydedilemedi!\n{e}")

    save_info = {
        "path": str(model_save_path),
        "model": model_parameters,
        "trainer": trainer_parameters,
    }

    save_info_path = Path(save_dir) / "saved_models.json"

    if not save_info_path.exists():
        save_info_path.write_text("{}")

    all_saved_models = dict()
    try:
        with open(save_info_path, "r", encoding="utf-8") as f:
            all_saved_models = json.load(f)
    except Exception as e:
        sys.exit(f"kayıt dosyası okunamadı!{e}")

    all_saved_models[name] = save_info

    try:
        with open(save_info_path, "w", encoding="utf-8") as f:
            json.dump(all_saved_models, f, indent=4, ensure_ascii=False)
    except Exception as e:
        sys.exit(f"kayıt dosyasına yazılamadı!\n{e}")


def load_model(name: str, save_dir: str) -> torch.nn.Module:
    save_info_path = Path(save_dir) / "saved_models.json"

    if not save_info_path.exists():
        sys.exit("kayıt içeriği bulunamadı!")

    try:
        with open(save_info_path, "r", encoding="utf-8") as f:
            loaded_info = json.load(f)
    except Exception as e:
        sys.exit(f"kayıt içeriği okunamadı!\n{e}")

    if name not in loaded_info.keys():
        sys.exit("bu isimde kayıtlı model bulunamadı")

    model_info = loaded_info[name]["model"]
    model_save_dir = loaded_info[name]["path"]

    model: torch.nn.Module = build_model(model_info)

    model.load_state_dict(torch.load(f=model_save_dir))

    return model


def is_exist(save_dir: str, model_conf: dict, trainer_conf: dict):
    save_info_path = Path(save_dir) / "saved_models.json"

    if not save_info_path.exists():
        save_info_path.write_text("{}")

    try:
        with open(save_info_path, "r", encoding="utf-8") as f:
            saved_models = json.load(f)
    except Exception as e:
        sys.exit(f"kayıt içeriği okunamadı!\n{e}")

    for saved_name, saved_conf in saved_models.items():
        saved_model_conf = saved_conf["model"]
        saved_trainer_conf = saved_conf["trainer"]

        if saved_model_conf == model_conf and saved_trainer_conf == trainer_conf:
            return True
    return False
