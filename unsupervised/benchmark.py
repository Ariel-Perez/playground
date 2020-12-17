import argparse
import contextlib
import enum
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import dataset
import unsupervised.algorithm as algorithm


class Model(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        # training_step defined the train loop. It is independent of forward
        x, _ = batch
        o = self(x)
        loss = self.loss(o, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        o = self(x)
        loss = self.loss(o, x)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, _):
        x, _ = batch
        o = self(x)
        loss = self.loss(o, x)
        self.log('test_loss', loss)
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # return [optimizer], [scheduler]
        return optimizer


class Callback(pl.callbacks.Callback):

    def on_train_epoch_end(self, trainer):
        print('Epoch end')


def train_and_evaluate(algo, dset, augment=False, debug=False, **kwargs):
    train, test, dimensions, _ = dataset.train_test(dset, augment=augment)
    if algo == algorithm.Algorithm.AUTO_ENCODER:
        model = algorithm.AutoEncoder(
            dimensions,
            embedding_dim=128,
            hidden_layers=[32, 64, 128, 256],
        )
    else:
        raise NotImplementedError('Algorithm not implemented: %s' % algo.name)

    lightning_model = Model(model)
    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[

    ])
    train_loader = data.DataLoader(train, **kwargs)
    test_loader = data.DataLoader(test, batch_size=1024)

    torch.set_printoptions(precision=4, sci_mode=False)
    context = torch.autograd.detect_anomaly() if debug else contextlib.suppress()
    with context:
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
