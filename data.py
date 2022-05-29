from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def train_test_dataloader(train_data, test_data, batch_size ):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader

def fashion_mnist_train_test_dataloader(batch_size):
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )
    return  train_test_dataloader(train_data, test_data, batch_size)
