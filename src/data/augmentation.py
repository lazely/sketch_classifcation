import yaml
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightnessContrast, HueSaturationValue, 
    GaussNoise, Blur, OpticalDistortion, GridDistortion, 
    ElasticTransform, CoarseDropout
)

def get_augmentation(config):
    aug_dict = config['training']['augmentation']
    aug_ops = []

    for aug_name, aug_prob in aug_dict.items():
        if aug_name == 'crop':
            aug_ops.append(RandomCrop(height=224, width=224, p=aug_prob))
        elif aug_name == 'flip':
            aug_ops.append(HorizontalFlip(p=aug_prob))
            aug_ops.append(VerticalFlip(p=aug_prob))
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

    return Compose(aug_ops)

def apply_augmentation(image, augmentation):
    return augmentation(image=image)['image']

# YAML 설정 파일을 로드하는 함수
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)