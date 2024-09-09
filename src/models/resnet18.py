from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.nn as nn

def get_model(config):
    model_config = config['model']
    
    # Hugging Face에서 ResNet 모델 불러오기
    model = ResNetForImageClassification.from_pretrained(f"microsoft/{model_config['name']}")
    
    # 마지막 분류 층만 수정 -> microsoft/resnet-18은 imagenet으로 pretrain된거라 필요 없긴 함
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, model_config['num_classes'])
    
    return model

def get_feature_extractor(config):
    model_config = config['model']
    return AutoImageProcessor.from_pretrained(f"microsoft/{model_config['name']}")