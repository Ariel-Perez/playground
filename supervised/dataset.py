import enum
import torchvision


class Dataset(enum.Enum):
    MNIST = 0
    FASHION_MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3
    IMAGENET = 4



def image_augmentation(dimensions):
    return [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(
            size=dimensions[0],
            padding=4)
    ]


def train_test(dataset: Dataset, augment: bool = False):
    augmentation = None
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
    elif dataset == Dataset.IMAGENET:
        constructor = torchvision.datasets.ImageNet
        dimensions = (224, 224, 3)
        labels = 1000
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)

    tensor_transform = torchvision.transforms.ToTensor()
    if augment:
        transform = torchvision.transforms.Compose(
            (augmentation or image_augmentation(dimensions)) + [tensor_transform])
    else:
        transform = tensor_transform
    train = constructor('data', train=True, download=True, transform=transform)
    test = constructor('data', train=False, download=True, transform=transform)
    return train, test, dimensions, labels
