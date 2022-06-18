# Tran Nhiem -- 2022/05
from typing import Optional, Sequence
import torchvision
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.transforms import autoaugment as auto_aug
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path


class DataModule_lightning(pl.LightningDataModule):

    def __init__(self, data_dir: str, dataset_name: str, img_size: int, batch_size: int, num_workers: int, dataloader_type: str,
                 dataset_urls: Optional[str] = None, download_dataset_url: Optional[bool] = None,
                 dataset_std: Optional[Sequence[float]] = None, dataset_mean: Optional[Sequence[float]] = None,
                 RandAug: Optional[Sequence[int]] = None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_name = dataset_name
        self.dataloader_type = dataloader_type
        self.dataset_urls = dataset_urls
        self.download_dataset_url = download_dataset_url
        self.resize_img = img_size
        # self.mean = dataset_mean,
        # self.std = dataset_std,
        self.RandAug =[1,9] #RandAug,

        self.dataset_transforms = {
            "classification":{
            "train": self.train_transform,
            "val": self.val_transform,
            "test": self.test_transform,
            }
        }
        if dataset_std :
            print("Using Custome Dataset (Mean, STD)")
            self.mean = dataset_mean,
            self.std = dataset_std,
        else: 
            print("Using ImageNet Dataset (Mean, STD)")
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.228, 0.224, 0.225)

    def prepare_dataset(self):
        if self.download_dataset_url:
            download_and_extract_archive(self.dataset_urls, self.data_dir)

    def Standard_Dataloader(self, mode: str):
        is_train = True if mode == "train" else False

        if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            # Initial Mean & STD for dataset
           
            mean=(0.4914, 0.4822, 0.4465) 
            std=(0.2023, 0.1994, 0.2010)
            print("Using Torch Train and Val")
            datapath = self.data_path.joinpath('dataset')

            # load Cifar10 dataset
            if self.dataset_name == "CIFAR10":
                train_set = datasets.CIFAR10(
                    root=datapath.joinpath(mode), train=True,
                    download=True, transform=self.dataset_transforms[mode],
                )

                val_set = datasets.CIFAR10(datapath.joinpath(mode), train=False,
                                           download=False, transform=self.dataset_transforms[mode],)
                if mode == "train":
                    return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                elif mode == "val":
                    return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)

            # Load Cifar100 Dataset
            if self.dataset_name == "CIFAR100":
                train_transform=transforms.Compose(
                    [ 
                    transforms.Resize( (self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                    auto_aug.RandAugment(num_ops=self.RandAug[0], magnitude=self.RandAug[1]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
                )
                train_set = datasets.CIFAR100(
                    root=datapath.joinpath(mode), train=True,
                    download=True, transform=train_transform,)
                
                val_transform=transforms.Compose(
                            [
                                transforms.Resize((self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                            ]
                        )

                val_set = datasets.CIFAR100(datapath.joinpath(mode), train=False,
                                            download=True, transform=val_transform,)
                if mode == "train":

                    return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
                elif mode == "val":

                    return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)

        else:

            if mode=='train': 
                transform=self.train_transform
            elif mode=='val': 
                transform=self.val_transform
            elif mode=='test': 
                transform= self.test_transform
            
            data_dir = self.data_path
            #transforms_=self.dataset_transforms["classification"][mode]
            dataset = ImageFolder(data_dir.joinpath(mode),transform)
            return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)

    def ffcv_Dataloader(self, mode: str):
        pass

    def train_dataloader(self):
        if self.dataloader_type == "ffcv_loader":
            return self.ffcv_Dataloader(mode="train")
        else:
            return self.Standard_Dataloader(mode="train")

    def val_dataloader(self):
        if self.dataloader_type == "ffcv_loader":
            return self.ffcv_Dataloader(mode="val")
        else:
            return self.Standard_Dataloader(mode="val")

    def test_dataloader(self):
        if self.dataloader_type == "ffcv_loader":
            return self.ffcv_Dataloader(mode="test")
        else:
            return self.Standard_Dataloader(mode="test")

    @property
    def train_transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        if self.RandAug is list:
            print("RandAugmentation is Implemented")
            return transforms.Compose(
                [
                    transforms.Resize(
                        (self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                    auto_aug.RandAugment(num_ops=self.RandAug[0], magnitude=self.RandAug[1]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(
                        (self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )

    @property
    def val_transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize((self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

    @property
    def test_transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.resize_img, self.resize_img), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

    @property
    def data_path(self):
        return Path(self.data_dir)

if __name__=="__main__": 
    print("Hoooray ~ You are testing the lightning DATA_Module ~")
    print("Oh wait you are not define any testing module yet")
