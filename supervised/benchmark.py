import argparse
import enum
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from supervised import algorithm
from supervised import dataset


class Model(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        accuracy = (pred == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)
        return {'val_loss': loss, 'val_acc': accuracy}

    def test_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        accuracy = (pred == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        return {'test_loss': loss, 'test_acc': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=3e-4)
        return optimizer


def train_and_evaluate(algo, dset, **kwargs):
    train, test, dimensions, labels = dataset.train_test(dset)
    if algo == algorithm.Algorithm.LINEAR:
        model = algorithm.Linear(dimensions, labels)
    elif algo == algorithm.Algorithm.DNN:
        model = algorithm.DNN(dimensions, labels)
    elif algo == algorithm.Algorithm.CNN:
        model = algorithm.CNN(dimensions, labels)
    elif algo == algorithm.Algorithm.SLIM:
        model = algorithm.Slim(dimensions, labels)
    else:
        raise NotImplementedError('Algorithm not implemented: %s' % algo.name)

    lightning_model = Model(model)
    trainer = pl.Trainer(gpus=1, precision=16)
    train_loader = data.DataLoader(train, **kwargs)
    test_loader = data.DataLoader(test, batch_size=1024)

    trainer.fit(lightning_model, train_loader)
    trainer.test(lightning_model, test_loader)
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
        batch_size=256, num_workers=4,
        pin_memory=True, shuffle=True)
