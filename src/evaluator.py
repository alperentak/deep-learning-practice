import json
import sys
from datetime import datetime
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


class Evaluator:
    def __init__(
        self,
        loss_fn: torch.nn.Module,
        device: str,
    ):
        self.device: str = device
        self.loss_fn: torch.nn.Module = loss_fn.to(device)

    def eval_model(
        self,
        model: torch.nn.Module,
        test_data_loader: DataLoader,
        model_name: str,
        eval_results_path: str,
    ):
        """
        modeli değerlendirir ve metrik değerlerini kaydeder
        """
        accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(self.device)

        total_loss: float = 0.0
        total_acc: float = 0.0

        model = model.to(self.device)
        model.eval()
        with torch.inference_mode():
            for X_test, y_test in test_data_loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)

                # forward pass
                y_pred = model(X_test)

                # calculate loss and accuracy
                loss: torch.Tensor = self.loss_fn(y_pred, y_test)
                total_loss += loss.item()
                acc: torch.Tensor = accuracy_fn(y_pred, y_test)
                total_acc += acc.item()

        avg_loss = total_loss / len(test_data_loader)
        avg_acc = total_acc / len(test_data_loader)

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }

        eval_results_json: Path = Path(eval_results_path)

        if not eval_results_json.exists():
            eval_results_json.write_text("{}")

        try:
            with open(eval_results_json, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
        except Exception as e:
            sys.exit(f"{eval_results_json} dosyası okunamadı!\n{e}")

        eval_name = model_name + "_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        eval_results[eval_name] = metrics

        try:
            with open(eval_results_json, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            sys.exit(f"{eval_results_json} dosyasına yazılamadı!\n{e}")

        click.echo(
            f"{eval_name}: eval değerleri {eval_results_json} adresine kaydedildi"
        )
