import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from typing import Union, Tuple, Callable
from src.data.augmentation import get_augmentation, apply_augmentation
import numpy as np

class AugmentationWrapper:
    def __init__(self, augmentation):
        self.augmentation = augmentation

    def __call__(self, img):
        # Convert PIL image to numpy array
        np_img = np.array(img)
        
        # Apply each augmentation sequentially
        for aug in self.augmentation:
            np_img = apply_augmentation(np_img, aug)
        
        # Convert back to PIL Image after augmentations
        return transforms.ToPILImage()(np_img)

def get_transform(config, is_train=True):
    if is_train:
        augmentation = get_augmentation(config['training'])
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            AugmentationWrapper(augmentation),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_file: str, 
        augmented_dir: str,
        augmented_info_file: str,
        transform: Callable,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.is_inference = is_inference
        
        # Read original data
        self.info_df = pd.read_csv(info_file)
        self.image_paths = self.info_df['image_path'].tolist()
        
        # Read augmented data
        self.augmented_df = pd.read_csv(augmented_info_file)
        self.augmented_image_paths = self.augmented_df['image_path'].tolist()

        # Combine original and augmented data
        if self.is_inference:
            self.all_image_paths = self.image_paths
        else:
            self.all_image_paths = self.image_paths + self.augmented_image_paths
        
        if not self.is_inference:
            self.targets = self.info_df['target'].tolist() + self.augmented_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        if index < len(self.image_paths):
            img_path = os.path.join(self.root_dir, self.all_image_paths[index])
        else:
            img_path = os.path.join(self.augmented_dir, self.all_image_paths[index])
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target

        
def get_test_loaders(config):
    dataset = CustomDataset(
        root_dir=config['data']['test_dir'],
        info_file=config['data']['test_info_file'],
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config,is_train=False),
        is_inference=True
    )
    test_data_loaders = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    return test_data_loaders

def get_data_loaders(config):
    train_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config)
    )

    train_size = int((1-config['training']['validation_ratio']) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def get_kfold_loaders(config: dict, n_splits: int = 5):
    dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config)
    )
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        yield fold, train_loader, val_loader

def get_inference_loader(config: dict):
    dataset = CustomDataset(
        root_dir=config['data']['test_dir'],
        info_file=config['data']['test_info_file'],
        transform=get_transform(),
        is_inference=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader