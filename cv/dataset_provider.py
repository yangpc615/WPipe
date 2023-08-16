import os
import torch
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cv.imagenet_dataset import ImageNetData

dataset_func_dict = {}


def get_dataloader_func(dataset_name):
    if dataset_name in dataset_func_dict:
        return dataset_func_dict[dataset_name]


def get_dataloader_by_imagedir(batch_size,
                               num_workers,
                               traindir=None,
                               valdir=None,
                               no_dataloader=False):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = val_loader = train_sampler = train_dataset = val_dataset = None
    if traindir is not None:
        traindir = os.path.join(traindir, 'train')

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler)

    if valdir is not None:
        valdir = os.path.join(valdir, 'val')
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
    if no_dataloader:
        return train_dataset, val_dataset
    return train_loader, val_loader, train_sampler


def get_oxford_flowers102_dataloader(batch_size,
                                     num_workers,
                                     traindir=None,
                                     valdir=None,
                                     no_dataloader=False):
    return get_dataloader_by_imagedir(batch_size,
                                      num_workers,
                                      traindir=traindir,
                                      valdir=valdir,
                                      no_dataloader=no_dataloader)


def get_cifar_dataloader(dataset_name,
                         batch_size,
                         num_workers,
                         traindir=None,
                         valdir=None,
                         no_dataloader=False):
    CifarDataset = CIFAR10 if dataset_name == "cifar10" else CIFAR100
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = val_loader = train_sampler = train_dataset = val_dataset = None
    if traindir is not None:
        train_dataset = CifarDataset(traindir,
                                     True,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))

        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler)

    if valdir is not None:
        val_dataset = CifarDataset(valdir,
                                   False,
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
    if no_dataloader:
        return train_dataset, val_dataset
    return train_loader, val_loader, train_sampler


def get_cifar10_dataloader(batch_size,
                           num_workers,
                           traindir=None,
                           valdir=None,
                           no_dataloader=False):
    return get_cifar_dataloader("cifar10",
                                batch_size,
                                num_workers,
                                traindir=traindir,
                                valdir=valdir,
                                no_dataloader=no_dataloader)


def get_cifar100_dataloader(batch_size,
                            num_workers,
                            traindir=None,
                            valdir=None,
                            no_dataloader=False):
    return get_cifar_dataloader("cifar100",
                                batch_size,
                                num_workers,
                                traindir=traindir,
                                valdir=valdir,
                                no_dataloader=no_dataloader)


def get_imagenet_dataloader(batch_size,
                            num_workers,
                            traindir=None,
                            valdir=None,
                            no_dataloader=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = val_loader = train_sampler = train_dataset = val_dataset = None
    if traindir is not None:
        train_dataset = ImageNetData(traindir,
                                     'train',
                                     'train_meta.txt',
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler)

    if valdir is not None:
        val_dataset = ImageNetData(valdir,
                                   "val",
                                   "val_meta.txt",
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
    if no_dataloader:
        return train_dataset, val_dataset
    return train_loader, val_loader, train_sampler


dataset_func_dict["imagenet"] = get_imagenet_dataloader
dataset_func_dict["cifar10"] = get_cifar10_dataloader
dataset_func_dict["cifar100"] = get_cifar100_dataloader
dataset_func_dict["oxfordflowers102"] = get_oxford_flowers102_dataloader
