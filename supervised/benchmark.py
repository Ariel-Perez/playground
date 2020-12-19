import argparse
import contextlib
import enum
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import dataset
import supervised.algorithm as algorithm


def train_and_evaluate(algo, dataset_name, augment=False, debug=False, **kwargs):
    dset = dataset.Data.create(dataset_name, augment=augment)
    train, test = dset.train(), dset.test()
    sample_x = train[0][0]
    dimensions = sample_x.shape
    labels = dset.num_labels
    if algo == algorithm.Algorithm.LINEAR:
        model = algorithm.Linear(dimensions, labels)
    elif algo == algorithm.Algorithm.DNN:
        model = algorithm.DNN(dimensions, labels, num_layers=10)
    elif algo == algorithm.Algorithm.CNN:
        model = algorithm.CNN(dimensions, labels)
    elif algo == algorithm.Algorithm.HIGHWAY_NETWORK:
        model = algorithm.HighwayNetwork(dimensions, labels, num_layers=100)
    elif algo == algorithm.Algorithm.RESNET:
        model = algorithm.ResidualNetwork(dimensions, labels, n=3)
    else:
        raise NotImplementedError('Algorithm not implemented: %s' % algo.name)

    trainer = pl.Trainer(gpus=1, precision=16)
    train_loader = data.DataLoader(train, **kwargs)
    test_loader = data.DataLoader(test, batch_size=1024)

    torch.set_printoptions(precision=4, sci_mode=False)
    context = torch.autograd.detect_anomaly() if debug else contextlib.suppress()
    with context:
        trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    return model


if __name__ == '__main__':
    dataset_names = [d.name for d in dataset.Dataset]
    algorithm_names = [a.name for a in algorithm.Algorithm]
    parser = argparse.ArgumentParser(description='Download datasets.')
    parser.add_argument(
        'dataset',
        help='Name of the dataset to use. '
             'Available options are: %s' % dataset_names)
    parser.add_argument(
        'algorithm',
        help='Name of the algorithm to use. '
             'Available options are: %s' % algorithm_names)
    parser.add_argument(
        '--augment', action='store_true',
        help='Whether to perform image augmentation')
    parser.add_argument(
        '--debug', action='store_true',
        help='Whether to enable gradient anomaly detection')

    args = parser.parse_args()
    try:
        dset = dataset.Dataset[args.dataset.upper()]
    except KeyError:
        raise KeyError('Invalid dataset (%s), must be in %s' %
                       (args.dataset, dataset_names))

    try:
        algo = algorithm.Algorithm[args.algorithm.upper()]
    except KeyError:
        raise KeyError('Invalid algorithm (%s), must be in %s' %
                       (args.algorithm, algorithm_names))

    train_and_evaluate(
        algo, dset,
        augment=args.augment, debug=args.debug,
        batch_size=128, num_workers=4,
        pin_memory=True, shuffle=True)
