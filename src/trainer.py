from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: str,
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.epochs = epochs
        self.device = device

    def train_model(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
    ) -> list[dict]:
        """
        model'i eğitir ve metrik geçmişi ile geçen süreyi döndürür
        """

        model.to(self.device)

        accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(self.device)

        history = list()

        timer_start = timer()

        model.train()
        for epoch in tqdm(range(self.epochs)):
            for batch, (X_train, y_train) in enumerate(train_data_loader):
                batch_start = timer()

                X_train, y_train = (
                    X_train.to(self.device, non_blocking=True),
                    y_train.to(self.device, non_blocking=True),
                )

                # forward pass
                y_pred = model(X_train)

                # calculate the loss and accuracy
                loss: torch.Tensor = self.loss_fn(y_pred, y_train)
                acc = accuracy_fn(y_pred, y_train)

                # optimizer zero grad
                self.optimizer.zero_grad()

                # loss backward
                loss.backward()

                # optimizer step
                self.optimizer.step()

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
