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


def denormalization(dataset: Dataset):
    norm = normalization(dataset)
    return torchvision.transforms.Normalize(
        mean=tuple(-u / s for u, s in zip(norm.mean, norm.std)),
        std=tuple(1 / s for s in norm.std),
    )


def normalization(dataset: Dataset):
    if dataset in [Dataset.MNIST, Dataset.FASHION_MNIST]:
        return torchvision.transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        )
    elif dataset in [Dataset.CIFAR10, Dataset.CIFAR100]:
        return torchvision.transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(1.0, 1.0, 1.0,),
        )
    elif dataset == Dataset.IMAGENET:
        return torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)


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
        ]
    else:
        raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)

    test_transforms = [torchvision.transforms.ToTensor()]
    train_transforms = [torchvision.transforms.ToTensor()]
    if augment:
        train_transforms.extend(augmentations)
    train_transforms.append(normalization(dataset))
    test_transforms.append(normalization(dataset))
    train = constructor(
        'data', train=True, download=True,
        transform=torchvision.transforms.Compose(train_transforms))
    test = constructor(
        'data', train=False, download=True,
        transform=torchvision.transforms.Compose(test_transforms))
    return train, test, dimensions, labels
