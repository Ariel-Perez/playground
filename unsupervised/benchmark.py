import argparse
import contextlib
import enum
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as utils

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
        return loss

    def test_step(self, batch, _):
        x, _ = batch
        o = self(x)
        loss = self.loss(o, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # return [optimizer], [scheduler]
        return optimizer


class Visualize(pl.callbacks.Callback):

    def __init__(self, dataloader, output_path, denormalization):
        os.makedirs(output_path, exist_ok=True)
        self.dataloader = dataloader
        self.output_path = output_path
        self.denormalization = denormalization
        self.epoch = 0

    def on_train_epoch_start(self, trainer, model):
        print(' Writing sample images to %s' % self.output_path)
        x, _ = next(iter(self.dataloader))
        x = x.cuda()
        out = model(x)

        out_split = torch.split(out, 1, dim=0)
        out_tensor = torch.cat(out_split, dim=2)
        x_split = torch.split(x, 1, dim=0)
        x_tensor = torch.cat(x_split, dim=2)

        final_tensor = self.denormalization(
            torch.cat([x_tensor, out_tensor], dim=3))
        if final_tensor.shape[1] == 1:
            final_tensor = final_tensor.repeat(1, 3, 1, 1)

        utils.save_image(final_tensor.squeeze(),
                         os.path.join(
                             self.output_path,
                             'epoch_%i.jpg' % self.epoch))
        self.epoch += 1


def train_and_evaluate(algo, dataset_name, augment=False, debug=False, **kwargs):
    dset = dataset.Data.create(dataset_name, augment=augment)
    train, test = dset.train(), dset.test()
    dimensions = train[0][0].shape
    if algo == algorithm.Algorithm.AUTOENCODER:
        model = algorithm.AutoEncoder(
            dimensions,
            embedding_dim=256,
            hidden_layers=[32, 64, 128, 256],
        )
    elif algo == algorithm.Algorithm.VAE:
        model = algorithm.VariationalAutoEncoder(
            dimensions,
            embedding_dim=256,
            hidden_layers=[32, 64, 128, 256],
        )
    else:
        raise NotImplementedError('Algorithm not implemented: %s' % algo.name)

    lightning_model = Model(model)
    train_loader = data.DataLoader(train, **kwargs)
    val_loader = data.DataLoader(test, batch_size=16)
    test_loader = data.DataLoader(test, batch_size=1024)
    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[
        Visualize(val_loader, 'save/unsupervised/%s' % dataset_name.name,
                  dset.denormalization),
    ])

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
