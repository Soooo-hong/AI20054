import os 

import torch 
import torchvision
import torch.nn.functional as F 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10,MNIST
from torchvision import transforms

from model import MNIST_model,CIFAR_model

mnist_model = MNIST_model()

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

mnist_model = MNIST_model()

train_ds = MNIST("./datasets", train=True, download=True, transform=transforms.ToTensor())
test_ds = MNIST("./datasets", train=False, download=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = pl.Trainer(
    accelerator='auto',
    devices=1,
    max_epochs=30,
    logger=CSVLogger(save_dir="MNIST_model/"),
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
trainer.test(mnist_model,dataloaders = test_dataloader)


pl.seed_everything(7)

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
cifar10_normalization = torchvision.transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)


def split_dataset(dataset, val_split=0.2, train=True):
    """Splits the dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))

    if train:
        return dataset_train
    return dataset_val


def get_splits(len_dataset, val_split):
    """Computes split lengths for train and validation set."""
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    else:
        raise ValueError(f"Unsupported type {type(val_split)}")

    return splits


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization,
    ]
)
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization,
    ]
)

dataset_train = CIFAR10("./datasets", train=True, download=True, transform=train_transforms) # download시 True로 변경해야됨 
dataset_val = CIFAR10("./datasets", train=True, download=True, transform=test_transforms)
dataset_train = split_dataset(dataset_train)
dataset_val = split_dataset(dataset_val, train=False)
dataset_test = CIFAR10("./datasets", train=False, download=True, transform=test_transforms)

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

cifar_model = CIFAR_model(BATCH_SIZE=BATCH_SIZE, lr=0.05)

trainer = pl.Trainer( max_epochs=30,
    accelerator="auto",
    devices=1,
    logger=CSVLogger(save_dir="cifar10_model/"),
    callbacks=[LearningRateMonitor(logging_interval="step")],)
trainer.fit(cifar_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(cifar_model, dataloaders=test_dataloader)

