import enum
import torchvision


class Dataset(enum.Enum):
    MNIST = 0
    FASHION_MNIST = 1


def train_test(dataset: Dataset):
    transform = torchvision.transforms.ToTensor()
    if dataset == Dataset.MNIST:
        train = torchvision.datasets.MNIST(
            'data',
            train=True,
            download=True,
            transform=transform,
        )
        test = torchvision.datasets.MNIST(
            'data',
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == Dataset.FASHION_MNIST:
        train = torchvision.datasets.FashionMNIST(
            'data',
            train=True,
            download=True,
            transform=transform,
        )
        test = torchvision.datasets.FashionMNIST(
            'data',
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)

    return train, test
