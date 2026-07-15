from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    def __init__(
        self,
        classes: dict,
        data_list: list[dict],
        transform,
    ):
        self.classes = classes
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        item_path = self.data_list[index]["path"]
        label = self.data_list[index]["label"]

        item = Image.open(item_path).convert("RGB")

        if self.transform:
            return self.transform(item), label

        return item, label


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
