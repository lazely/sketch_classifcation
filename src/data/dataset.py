import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

class SketchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image'].convert('RGB'))
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return {'pixel_values': image, 'label': label}

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_full_dataset(config):
    # 데이터셋 로드
    raw_dataset = load_dataset(config['data']['data_dir'], trust_remote_code=True)['train']
    transform = get_transform()

    # SketchDataset 인스턴스 생성
    full_dataset = SketchDataset(raw_dataset, transform=transform)

    return full_dataset

def get_data_loaders(config):
    # 데이터셋 로드
    raw_dataset = load_dataset(config['data']['data_dir'])['train']
    transform = get_transform()

    # 전체 데이터셋 생성
    full_dataset = SketchDataset(raw_dataset, transform=transform)

    # 학습 및 검증 데이터셋 분할
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader