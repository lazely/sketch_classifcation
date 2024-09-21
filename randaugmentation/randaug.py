import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import *
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import random
from torchvision import transforms
from torchvision.transforms import RandAugment

def get_randaugment(config, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_ops = config['num_ops']
    magnitude = config['magnitude']
    return RandAugment(num_ops=num_ops, magnitude=magnitude)

def apply_augmentation(image, randaugment):
    # 이미지를 Tensor로 변환 후 RandAugment 적용
    image_tensor = transforms.ToTensor()(image)  # 0-1 범위의 float로 변환
    image_tensor = (image_tensor * 255).byte()  # 0-255 범위의 uint8로 변환
    augmented_image_tensor = randaugment(image_tensor)
    return transforms.ToPILImage()(augmented_image_tensor)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_augmented_image_name(original_name, aug_type):
    name, ext = os.path.splitext(original_name)
    return f"{name}_aug_{aug_type}{ext}"

def perform_offline_augmentation(config_path):
    
    train_dir = config['data']['train_dir']
    train_info_file = config['data']['train_info_file']
    augmented_dir = config['data']['augmented_dir']
    augmented_info_file = config['data']['augmented_info_file']
    
    # Randaugment 인스턴스 생성
    randaugment = get_randaugment(config['randaugment'])
    
    df = pd.read_csv(train_info_file)
    
    new_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting images"):
        relative_path = row['image_path']
        full_image_path = os.path.join(train_dir, relative_path)
        target = row['target']
        
        class_name = os.path.dirname(relative_path)
        augmented_class_dir = os.path.join(augmented_dir, class_name)
        os.makedirs(augmented_class_dir, exist_ok=True)
        
        image = cv2.imread(full_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply RandAugment
        augmented_image = apply_augmentation(image, randaugment)
        
        original_filename = os.path.basename(full_image_path)
        new_file_name = create_augmented_image_name(original_filename, "randaugment")
        new_relative_path = os.path.join(class_name, new_file_name)
        new_full_path = os.path.join(augmented_dir, new_relative_path)
        
        cv2.imwrite(new_full_path, cv2.cvtColor(np.array(augmented_image), cv2.COLOR_RGB2BGR))
        
        new_rows.append({
            'image_path': new_relative_path,
            'target': target,
            'augmentation': 'RandAugment'  # 로그 기록
        })
    
    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv(augmented_info_file, index=False)
    
    print(f"Augmentation complete. Number of augmented images: {len(augmented_df)}")

if __name__ == "__main__":
    config = get_config()
    perform_offline_augmentation(config, seed = 42)
