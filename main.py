import torch

from src.dataset import data_loader_info, load_fashion_mnist
from src.eval import eval_model
from src.models import BaselineModel, CNNModel
from src.trainer import Trainer


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_data_loader, test_data_loader, class_names = load_fashion_mnist(
        batch_size=128
    )

    data_loader_info(train_data_loader)
    data_loader_info(test_data_loader)

    model_0 = BaselineModel(in_shape=784, hidden_units=10, out_shape=len(class_names))
    model_1 = CNNModel(in_shape=1, hidden_units=10, out_shape=len(class_names))

    optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        device=device,
        epochs=50,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    train_history = trainer.train_model(
        model=model_1, train_data_loader=train_data_loader
    )

    for i in train_history:
        if i["epoch"] % 5 == 0 and i["batch"] == 0:
            print(i)

    y_pred, test_metrics = eval_model(
        model=model_1, test_data_loader=test_data_loader, device=device, loss_fn=loss_fn
    )

    print(y_pred)
    print("metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
