from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm


def train_model(
    model: nn.Module,
    train_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int = 200,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict]:
    """
    model'i eğitir ve metrik geçmişi ile geçen süreyi döndürür
    """

    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    loss_fn = loss_fn.to(device)
    model.to(device)

    accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    history = list()

    timer_start = timer()

    model.train()
    for epoch in tqdm(range(epochs)):
        for batch, (X_train, y_train) in enumerate(train_data_loader):
            batch_start = timer()

            X_train, y_train = (
                X_train.to(device, non_blocking=True),
                y_train.to(device, non_blocking=True),
            )

            # forward pass
            y_pred = model(X_train)

            # calculate the loss and accuracy
            loss: torch.Tensor = loss_fn(y_pred, y_train)
            acc = accuracy_fn(y_pred, y_train)

            # optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()

            # batch time and metrics
            batch_time = timer() - batch_start
            total_time = timer() - timer_start
            history.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "loss": loss.item(),
                    "accuracy": acc.item(),
                    "batch_time": batch_time,
                    "total_time": total_time,
                }
            )
    return history
