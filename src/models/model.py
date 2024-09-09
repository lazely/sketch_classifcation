from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification
import torch.nn as nn

def get_model(config):
    model_config = config['model']
    model = None
    if(model_config['name'] == "resnet"):
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    if(model_config['name'] == "ViT"):
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    # 마지막 분류 층만 수정
    #num_features = model.classifier[1].in_features
    #model.classifier[1] = nn.Linear(num_features, model_config['num_classes'])
    
    return model

def get_feature_extractor(config):
    model_config = config['model']
    model = None
    if(model_config['name'] == "resnet"):
        Extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    if(model_config['name'] == "ViT"):
        Extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return Extractor