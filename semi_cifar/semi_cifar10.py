import os
import numpy
import torch
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset


class SubsetCIFAR10(CIFAR10):
    def __init__(self, root, indices, **kwargs):
        super().__init__(root, **kwargs)
        assert len(indices) < len(self)
        self.data = self.data[indices]
        self.targets = torch.tensor(self.targets)[indices]


def index_select(labels, n, seed=None):
    labels = numpy.array(labels)
    classes = numpy.unique(labels)
    n_per_class = n // len(classes)
    random_state = numpy.random.RandomState(seed)

    labeled_indices = []
    for c in classes:
        class_indices = numpy.where(labels == c)[0]
        chosen_indices = random_state.choice(class_indices, size=n_per_class, replace=False)
        labeled_indices.extend(chosen_indices)
    return labeled_indices


class SemiCIFAR10(pl.LightningDataModule):

    def __init__(self, root, num_labeled=4000, batch_size=1, seed=None,
                 num_workers=None, pin_memory=True, expand_labeled=True):
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

        self.seed = None
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self.pin_memory = pin_memory
        self.expand_labeled = expand_labeled

    def prepare_data(self):
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        self.cifar10_valid = CIFAR10(self.root, train=False, transform=self.valid_transform)
        self.cifar10_trainᵤ = CIFAR10(self.root, train=True, transform=self.train_transformᵤ)

        indices = index_select(self.cifar10_trainᵤ.targets, self.num_labeled, self.seed)
        self.cifar10_trainₗ = SubsetCIFAR10(self.root, indices, train=True, transform=self.train_transformₗ)

        if self.expand_labeled:
            n = 1 + (len(self.cifar10_trainᵤ) - 1) // self.batch_sizeᵤ
            m = n * self.batch_sizeₗ // len(indices)
            self.cifar10_trainₗ = ConcatDataset([self.cifar10_trainₗ] * m)

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
    dm = SemiCIFAR10(**{
        "root": "data/cifar10",
        # "num_labeled": 10,
        # "num_labeled": 40,
        # "num_labeled": 250,
        # "num_labeled": 1000,
        "num_labeled": 4000,
        "batch_size": {
            "labeled": 64,
            "unlabeled": 448
        }
    })

    from torchvision import transforms
    transform = transforms.ToTensor()
    dm.train_transformₗ = transform
    dm.train_transformᵤ = transform
    dm.valid_transform = transform

    dm.prepare_data()
    dm.setup()

    print("dataset train-labeled:", len(dm.cifar10_trainₗ))
    print("dataset train-unlabeled:", len(dm.cifar10_trainᵤ))
    print("dataset valid:", len(dm.cifar10_valid))
    print("loader train-labeled:", len(dm.train_dataloader()['labeled']))
    print("loader train-unlabeled:", len(dm.train_dataloader()['unlabeled']))
    print("loader valid:", len(dm.val_dataloader()))
