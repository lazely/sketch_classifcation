from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification, ConvNextForImageClassification, EfficientNetForImageClassification
import timm
import torch.nn as nn

def get_model(config):
    model_config = config['model']
    model = None
    num_classes = model_config['num_classes']
    model_config_name = model_config['name']

    model_mapping = {
        "resnet":"resnet50",
        "ViT":"vit_base_patch16_224.augreg_in21k",
        "ConvN":"convnext_base",
        "eff3":"efficientnet_b3",
        "eff4":"efficientnet_b4",
        "eff5":"efficientnet_b5",
        "eff6":"efficientnet_b6",
        "eff7":"efficientnet_b7"
    }
    model_name = model_mapping[model_config_name]
    model = timm.create_model(model_name, pretrained=config['model']['pretrained'], num_classes=num_classes)

    # 모델 분류 층 수정
    if model:
        model = modify_classification_layer(model, num_classes)

    return model

def get_feature_extractor(config):
    model_config = config['model']
    model = None
    if(model_config['name'] == "resnet"):
        Extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    elif(model_config['name'] == "ViT"):
        Extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    elif(model_config['name'] == 'ConvN'):
        Extractor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    elif(model_config['name'] == 'eff3'):
        Extractor = AutoImageProcessor.from_pretrained("google/efficientnet-b3")
    elif(model_config['name'] == 'eff4'):
        Extractor = AutoImageProcessor.from_pretrained("google/efficientnet-b4")
    elif(model_config['name'] == 'eff5'):
        Extractor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
    elif(model_config['name'] == 'eff6'):
        Extractor = AutoImageProcessor.from_pretrained("google/efficientnet-b6")
    elif(model_config['name'] == 'eff7'):
        Extractor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
    return Extractor

def get_last_linear_layer(model):
    """
    모델에서 마지막 분류 층(nn.Linear)을 찾아 반환하는 함수.
    """
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = name  # 마지막으로 발견된 nn.Linear의 이름 저장
    return last_linear

def modify_classification_layer(model, num_classes):
    """
    모델의 마지막 분류 층을 num_classes에 맞게 수정하는 함수.
    """
    # 마지막 nn.Linear 레이어의 이름을 가져옴
    last_linear_name = get_last_linear_layer(model)
    
    # 마지막 레이어가 존재할 때 수정
    if last_linear_name:
        # 마지막 레이어 객체에 접근
        outclass = dict(model.named_modules())[last_linear_name]
        num_features = outclass.in_features
        
        # 마지막 분류 층을 수정하여 새로운 레이어로 변경
        new_layer = nn.Linear(num_features, num_classes)
        
        # 모델의 속성을 동적으로 업데이트
        *parents, last = last_linear_name.split(".")
        parent_module = model
        for attr in parents:
            parent_module = getattr(parent_module, attr)
        
        # 최종 부모 모듈에서 마지막 레이어를 수정
        setattr(parent_module, last, new_layer)
    
    return model