import os
import re
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from typing import Union, Tuple, Callable
from src.data.augmentation import get_augmentation, apply_augmentation
import numpy as np

def get_next_version(directory):
    """
    현재 디렉토리에서 가장 높은 버전 번호를 찾고, 그 다음 번호를 반환합니다.
    """
    version_pattern = re.compile(r'train_info(\d+)\.csv')
    highest_version = 0
    
    # 디렉토리의 모든 파일을 검색하여 버전 번호를 추출
    for filename in os.listdir(directory):
        match = version_pattern.match(filename)
        if match:
            version = int(match.group(1))
            if version > highest_version:
                highest_version = version
    
    # 다음 버전 번호를 반환
    return highest_version + 1

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
        use_augmented: bool = True,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.is_inference = is_inference
        self.use_augmented = use_augmented
        # Read original data
        self.info_df = pd.read_csv(info_file)
        self.image_paths = self.info_df['image_path'].tolist()
        if self.use_augmented:
            # Read augmented data
            self.augmented_df = pd.read_csv(augmented_info_file)
            self.augmented_image_paths = self.augmented_df['image_path'].tolist()
            self.all_image_paths = self.image_paths + self.augmented_image_paths
        else:
            self.all_image_paths = self.image_paths
        if not self.is_inference:
            if self.use_augmented:
                self.targets = self.info_df['target'].tolist() + self.augmented_df['target'].tolist()
            else:
                self.targets = self.info_df['target'].tolist()
    def __len__(self) -> int:
        return len(self.all_image_paths)
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        if self.use_augmented and index >= len(self.image_paths):
            img_path = os.path.join(self.augmented_dir, self.all_image_paths[index])
        else:
            img_path = os.path.join(self.root_dir, self.all_image_paths[index])
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
        is_inference=True,
        use_augmented=False
    )
    test_data_loaders = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        drop_last=False
    )
    return test_data_loaders

def get_data_loaders(config, batch_size=None):
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # 전체 데이터셋 로드 (오프라인 증강 이미지는 사용하지 않음)
    full_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        augmented_dir=None,  # 오프라인 증강 이미지 디렉토리 제외
        augmented_info_file=None,  # 오프라인 증강 정보 파일 제외
        transform=get_transform(config, is_train=False),  # 기본 Transform 적용 (증강 X)
        use_augmented=False  # 오프라인 증강 이미지를 사용하지 않음
    )
    
    # 8:2로 train/val 나누기
    train_size = int((1 - config['training']['validation_ratio']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # train/val 나눈 데이터 저장
    train_df = full_dataset.info_df.iloc[train_indices.indices]
    val_df = full_dataset.info_df.iloc[val_indices.indices]

    # 저장 경로 생성
    split_dir = config['data']['split_dir']
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    
    # 다음 버전 번호를 결정
    version = get_next_version(split_dir)
        
    train_save_path = os.path.join(split_dir, f'train_info{version}.csv')
    val_save_path = os.path.join(split_dir, f'val_info{version}.csv')
    
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    
    train_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=os.path.join(config['data']['split_dir'], f'train_info{version}.csv'),  # 버전 관리된 train 데이터
        augmented_dir=config['data']['augmented_dir'],  # 오프라인 증강 이미지 디렉토리 사용
        augmented_info_file=config['data']['augmented_info_file'],  # 오프라인 증강 정보 파일 사용
        transform=get_transform(config, is_train=True),  # 증강 적용
        use_augmented=True  # 오프라인 증강 이미지 포함
    )
    
    val_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=os.path.join(config['data']['split_dir'], f'val_info{version}.csv'),  # 버전 관리된 val 데이터
        augmented_dir=None,  # 오프라인 증강 이미지 미사용
        augmented_info_file=None,  # 오프라인 증강 정보 파일 미사용
        transform=get_transform(config, is_train=False),  # 증강 미적용
        use_augmented=False  # 오프라인 증강 미사용
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        transform=get_transform(config),
        use_augmented=False
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
        is_inference=True,
        use_augmented=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader