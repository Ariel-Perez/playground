import enum
import torchvision


class Dataset(enum.Enum):
    MNIST = 0
    FASHION_MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3


def train_test(dataset: Dataset):
    transform = torchvision.transforms.ToTensor()
    if dataset == Dataset.MNIST:
        constructor = torchvision.datasets.MNIST
        dimensions = (28, 28, 1)
        labels = 10
    elif dataset == Dataset.FASHION_MNIST:
        constructor = torchvision.datasets.FashionMNIST
        dimensions = (28, 28, 1)
        labels = 10
    elif dataset == Dataset.CIFAR10:
        constructor = torchvision.datasets.CIFAR10
        dimensions = (32, 32, 3)
        labels = 10
    elif dataset == Dataset.CIFAR100:
        constructor = torchvision.datasets.CIFAR100
        dimensions = (32, 32, 3)
        labels = 100
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)

    train = constructor('data', train=True, download=True, transform=transform)
    test = constructor('data', train=False, download=True, transform=transform)
    return train, test, dimensions, labels
