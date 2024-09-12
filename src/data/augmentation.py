import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from albumentations import (
    Compose, RandomCrop, CenterCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightnessContrast, HueSaturationValue, 
    GaussNoise, Blur, OpticalDistortion, GridDistortion, 
    ElasticTransform, CoarseDropout, Resize
)
def get_augmentation(config):
    aug_ops = []
    aug_dict = config['augmentation']
    
    for aug_name, aug_prob in aug_dict.items():
        if aug_name == 'crop':
            aug_ops.append(Compose([RandomCrop(height=150, width=150, p=aug_prob)]))
        elif aug_name == 'flip':
            aug_ops.append(Compose([HorizontalFlip(p=0.5),VerticalFlip(p=0.5)]))
        elif aug_name == 'brightness_contrast':
            aug_ops.append(RandomBrightnessContrast(p=aug_prob))
        elif aug_name == 'hue_saturation':
            aug_ops.append(HueSaturationValue(p=aug_prob))
        elif aug_name == 'noise':
            aug_ops.append(GaussNoise(p=aug_prob))
        elif aug_name == 'blur':
            aug_ops.append(Blur(blur_limit=7, p=aug_prob))
        elif aug_name == 'distortion':
            aug_ops.append(OpticalDistortion(p=aug_prob))
            aug_ops.append(GridDistortion(p=aug_prob))
            aug_ops.append(ElasticTransform(p=aug_prob))
        elif aug_name == 'mask':
            aug_ops.append(CoarseDropout(max_holes=8, max_height=32, max_width=32, p=aug_prob))

    return aug_ops

def apply_augmentation(image, augmentation):
    return augmentation(image=image)['image']

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
    
    # Create augmentation pipeline
    augmentations = get_augmentation(config['offline_augmentation'])
    
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
        
        for aug_type, aug in enumerate(augmentations):
            augmented_image = apply_augmentation(image, aug)
            
            original_filename = os.path.basename(full_image_path)
            new_file_name = create_augmented_image_name(original_filename, aug_type)
            new_relative_path = os.path.join(class_name, new_file_name)
            new_full_path = os.path.join(augmented_dir, new_relative_path)
            
            cv2.imwrite(new_full_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            
            new_rows.append({
                'image_path': new_relative_path,
                'target': target
            })
    
    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv(augmented_info_file, index=False)
    
    print(f"Augmentation complete. Number of augmented images: {len(augmented_df)}")

if __name__ == "__main__":
    config = get_config()

    perform_offline_augmentation(config)