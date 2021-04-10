import os
import numpy
import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10


class SmallCIFAR10(CIFAR10):
    def __init__(self, root, indices, **kwargs):
        super().__init__(root, **kwargs)
        assert len(indices) < len(self)
        self.data = self.data[indices]
        self.targets = torch.tensor(self.targets)[indices]


def index_select(labels, n, seed=None):
    labels = numpy.array(labels)
    classes = labels.unique()
    n_per_class = n // len(classes)
    random_state = numpy.random.RandomState(seed)

    indices = []
    for c in classes:
        indices.extend(random_state.choice(
            numpy.where(labels == c)[0],
            size=n_per_class,
            replace=False))
    return indices


class SemiCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_labeled=4000, batch_size=1,
                 num_workers=None, pin_memory=True, seed=None, expend=True):
        super().__init__()
        self.root = root
        self.num_labeled = num_labeled

        if isinstance(batch_size, dict):
            self.batch_sizeₗ = batch_size['labeled']
            self.batch_sizeᵤ = batch_size['unlabeled']
        elif isinstance(batch_size, int):
            self.batch_sizeₗ = batch_size
            self.batch_sizeᵤ = batch_size
        else:
            raise ValueError("batch_size should be 'int' or 'dict'")

        self.train_transformₗ = None
        self.train_transformᵤ = None
        self.valid_transform = None

        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.pin_memory = pin_memory
        self.seed = None
        self.expand = expand

    def prepare_data(self):
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_valid = CIFAR10(self.root, train=False, transform=self.valid_transform)
        self.cifar10_trainᵤ = CIFAR10(self.root, train=True, transform=self.train_transformᵤ)

        indices = index_select(self.cifar10_train.targets, self.num_labeled, self.seed)
        self.cifar10_trainₗ = SmallCIFAR10(self.root, indices, train=True, transform=self.train_transformₗ)

        if self.expand:
            n_iter = 1 + (len(self.cifar10_trainᵤ) - 1) // self.batch_sizeᵤ
            indices = indices * (n_iter * self.batch_sizeₗ // len(indices))


    def train_dataloader(self):
        loaderₗ = torch.utils.data.DataLoader(
            self.cifar10_trainₗ, self.batch_sizeₗ, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory)
        loaderᵤ = torch.utils.data.DataLoader(
            self.cifar10_trainᵤ, self.batch_sizeᵤ, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory)
        return {'labeled': loaderₗ, 'unlabeled': loaderᵤ}

    def val_dataloader(self):
        batch_size = max(self.batch_sizeₗ, self.batch_sizeᵤ) * 2
        return torch.utils.data.DataLoader(
            self.cifar10_valid, batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    from torchvision import transforms
    trfm = transforms.ToTensor()
    dm = SSL_CIFAR10(**{
        "root": "data/cifar10",
        "num_labeled": 4000,
        "batch_size": {
            "labeled": 64,
            "unlabeled": 448
        }
    })
    dm.train_transformₗ = trfm
    dm.train_transformᵤ = trfm
    dm.valid_transform = trfm
    dm.prepare_data()
    dm.setup()
    print("labeled dataset:", len(dm.cifar10_trainₗ))
    print("unlabeled dataset:", len(dm.cifar10_trainᵤ))
    print("test dataset:", len(dm.cifar10_test))
    print("labeled loader:", len(dm.train_dataloader()['labeled']))
    print("unlabeled loader:", len(dm.train_dataloader()['unlabeled']))
    print("test loader:", len(dm.val_dataloader()))
    #for x in zip(*dm.train_dataloader()):
    #    print(x)
    #for x in dm.val_dataloader():
    #    print(x)
    #l, u = dm.train_dataloader()
    #print(len(l), len(u))
