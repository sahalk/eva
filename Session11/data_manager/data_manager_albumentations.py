import torchvision
from torchvision import datasets

import torch
from albumentations import *
from albumentations.pytorch import ToTensor

class CIFAR10_dataset(datasets.CIFAR10):
    """
    Custom class to include albumentations data augmentations
    """

    def __init__(self, **kwargs):
        """
        Constructor for custom CIFAR10 dataset
        """
        super().__init__(**kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class DataManager(object):
    """
    Class that handles data management for an experiment
    """
    
    def __init__(self, batch_size=64, use_cuda=True, dataset_name='cifar10', 
                 trainset = None, testset = None,
                 train_transforms=None, test_transforms=None):
        super().__init__()

        self.dataset_name = dataset_name

        if self.dataset_name == 'cifar10':
            if trainset:
                self.trainset = trainset
            else:
                # Train Phase transformations
                if train_transforms:
                    self.train_transforms = train_transforms
                else:
                    self.train_transforms = train_transforms = Compose([Cutout(num_holes=4, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
                                                                        HorizontalFlip(p=0.5),
                                                                        Normalize(
                                                                            mean=[0.4914, 0.4822, 0.4465],
                                                                            std=[0.2471, 0.2435, 0.2616],
                                                                        ),
                                                                        ToTensor()
                                                                        ])
                self.trainset = CIFAR10_dataset(root='./data', train=True, download=True, transform=self.train_transforms)
            
            if testset:
                self.testset = testset
            else:
                # Test Phase transformations
                if test_transforms:
                    self.test_transforms = test_transforms
                else:
                    self.test_transforms = Compose([Normalize(
                                                        mean=[0.4914, 0.4822, 0.4465],
                                                        std=[0.2471, 0.2435, 0.2616],
                                                    ),
                                                    ToTensor()
                                                    ])
                self.testset = CIFAR10_dataset(root='./data', train=False, download=True, transform=self.test_transforms)

            # dataloader arguments - something you'll fetch these from cmdprmt
            dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=batch_size)

            # train dataloader
            self.train_loader = torch.utils.data.DataLoader(self.trainset, **dataloader_args)
            # test dataloader
            self.test_loader = torch.utils.data.DataLoader(self.testset, **dataloader_args)

    @staticmethod
    def padded_random_crop (x , **kwargs):
        x = PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=None, mask_value=None, always_apply=True).apply(x)
        x = RandomCrop(height=32, width=32, always_apply=True).apply(x)
        return x