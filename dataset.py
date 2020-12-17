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
    if dataset == Dataset.MNIST:
        constructor = torchvision.datasets.MNIST
        dimensions = (28, 28, 1)
        labels = 10
        augmentations = []
    elif dataset == Dataset.FASHION_MNIST:
        constructor = torchvision.datasets.FashionMNIST
        dimensions = (28, 28, 1)
        labels = 10
        augmentations = []
    elif dataset == Dataset.CIFAR10:
        constructor = torchvision.datasets.CIFAR10
        dimensions = (32, 32, 3)
        labels = 10
        augmentations = [image_augmentation(dimensions)]
    elif dataset == Dataset.CIFAR100:
        constructor = torchvision.datasets.CIFAR100
        dimensions = (32, 32, 3)
        labels = 100
        augmentations = [image_augmentation(dimensions)]
    elif dataset == Dataset.IMAGENET:
        constructor = torchvision.datasets.ImageNet
        dimensions = (224, 224, 3)
        labels = 1000
        augmentations = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomChoice([
                torchvision.transforms.Resize(256),
                torchvision.transforms.Resize(384),
                torchvision.transforms.Resize(480),
            ]),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)

    # std_transforms =
    tensor_transform = torchvision.transforms.ToTensor()
    augmentations.append(tensor_transform)
    if augment:
        transform = torchvision.transforms.Compose(augmentations)
    else:
        transform = tensor_transform
    train = constructor('data', train=True, download=True, transform=transform)
    test = constructor('data', train=False, download=True, transform=tensor_transform)
    return train, test, dimensions, labels
