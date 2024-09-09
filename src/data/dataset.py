from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

def get_data_loaders(config):
    # 데이터셋 로드
    dataset = load_dataset(config['data']['data_dir'])

    # 전처리 함수 정의
    def preprocess_function(examples):
        images = [image.convert("RGB") for image in examples['image']]
        examples['pixel_values'] = [transform(image) for image in images]
        return examples

    # 이미지 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋에 전처리 적용
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=['image'])
    dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    # 학습 및 검증 데이터셋 분할
    dataset = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader