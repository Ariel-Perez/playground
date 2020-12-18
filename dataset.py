import enum
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Dataset(enum.Enum):
    MNIST = 0
    FASHION_MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3
    IMAGENET = 4


class Data:
    def __init__(self, constructor, num_labels, normalization, augmentation=None):
        self.constructor = constructor
        self.num_labels = num_labels
        self.normalization = normalization
        self.denormalization = transforms.Normalize(
            mean=[-mean / std for mean, std in
                  zip(normalization.mean, normalization.std)],
            std=[1 / std for std in normalization.std],
        )
        self.tensorization = transforms.ToTensor()
        self.augmentation = augmentation

    def transform(self, augment=False):
        if augment:
            return transforms.Compose([
                self.tensorization,
                self.normalization,
                self.augmentation,
            ])
        return transforms.Compose([
            self.tensorization,
            self.normalization,
        ])

    def train(self):
        return self.constructor(
            'data', train=True, download=True,
            transform=self.transform(augment=self.augmentation)
        )

    def test(self):
        return self.constructor(
            'data', train=False, download=True,
            transform=self.transform(augment=False)
        )

    @classmethod
    def create(cls, dataset: Dataset, augment: bool = True):
        if dataset == Dataset.MNIST:
            return Data(
                datasets.MNIST,
                num_labels=10,
                normalization=transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            )
        elif dataset == Dataset.FASHION_MNIST:
            return Data(
                datasets.FashionMNIST,
                num_labels=10,
                normalization=transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            )
        elif dataset == Dataset.CIFAR10:
            return Data(
                datasets.CIFAR10,
                num_labels=10,
                normalization=transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(1.0, 1.0, 1.0,),
                ),
                augmentation=(transforms.RandomHorizontalFlip()
                              if augment else None),
            )
        elif dataset == Dataset.CIFAR100:
            return Data(
                datasets.CIFAR100,
                num_labels=100,
                normalization=transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(1.0, 1.0, 1.0,),
                ),
                augmentation=(transforms.RandomHorizontalFlip()
                              if augment else None),
            )
        elif dataset == Dataset.IMAGENET:
            return Data(
                datasets.ImageNet,
                num_labels=1000,
                normalization=transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                augmentation=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomChoice([
                        transforms.Resize(256),
                        transforms.Resize(384),
                        transforms.Resize(480),
                    ]),
                    transforms.RandomCrop(224),
                ]) if augment else None,
            )
        else:
            raise NotImplementedError('Dataset download not implemented: %s' % dataset.name)
