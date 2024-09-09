import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_config(config_name='default'):
    config_path = Path(__file__).parent.parent / 'configs' / f'{config_name}.yaml'
    config = load_config(config_path)
    
    if config['training']['device'] == 'cuda' and not torch.cuda.is_available():
        config['training']['device'] = 'cpu'
    
    return config

config = get_config()

def get_data_config():
    return config['data']

def get_model_config():
    return config['model']

def get_training_config():
    return config['training']

def get_paths_config():
    return config['paths']

def get_criterion(criterion_name):
    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    # 다른 손실 함수들을 여기에 추가할 수 있습니다.
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def get_optimizer(optimizer_config, model_parameters):
    optimizer_name = optimizer_config['name']
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=config['training']['learning_rate'], 
                          weight_decay=optimizer_config['weight_decay'])
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=config['training']['learning_rate'], 
                         momentum=0.9, weight_decay=optimizer_config['weight_decay'])
    # 다른 옵티마이저들을 여기에 추가할 수 있습니다.
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")