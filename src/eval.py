import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy


def eval_model(
    model: nn.Module, test_data_loader: DataLoader, device: str
) -> tuple[torch.Tensor, dict]:
    """
    Test verisi üzerinde değer tahmini yapar ve tahminleri ve metrik değerlerini döndürür
    """

    accuracy_fn = Accuracy(task="multiclass", num_classes=10)
    accuracy_fn = accuracy_fn.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    total_loss: float = 0
    total_acc: float = 0

    preds = list()

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward pass
            y_pred = model(X_test)
            preds.append(y_pred.cpu())

            # calculate loss and accuracy
            loss: torch.Tensor = loss_fn(y_pred, y_test)
            total_loss += loss.item()
            acc: torch.Tensor = accuracy_fn(y_pred, y_test)
            total_acc += acc.item()

    avg_loss = total_loss / len(test_data_loader)
    avg_acc = total_acc / len(test_data_loader)

    metrics = {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }

    preds = torch.cat(preds, dim=0)
    return preds, metrics
