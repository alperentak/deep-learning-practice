from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_fashion_mnist(batch_size: int = 32) -> tuple[DataLoader, DataLoader, list]:
    """
    FashionMNIST için train/test DataLoader'ları döndürür
    """
    train_data = datasets.FashionMNIST(
        train=True,
        root="data",
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        train=False, root="data", download=True, transform=ToTensor()
    )

    class_names = train_data.classes

    train_data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_data_loader, test_data_loader, class_names


def data_loader_info(data_loader: DataLoader):
    """
    DataLoader hakkında temel bilgileri yazdırır
    """

    features_batch, labels_batch = next(iter(data_loader))
    print("length\t| batch\t| features batch shape\t\t| labels batch shape")

    print(f" {len(data_loader)}", end="\t|")
    print(f" {data_loader.batch_size}", end="\t|")
    print(f" {features_batch.shape}", end="\t|")
    print(f" {labels_batch.shape}", end="")
    print()
