import optuna
import wandb
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import yaml

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from optuna.samplers import TPESampler

from src.config import *
from src.utils.metrics import get_metric_function
from scripts.train import train_one_epoch, validate, get_data_loaders
from src.data.dataset import CustomDataset, get_transform
from src.models.model import get_model

config = get_config()

hp_config = config['hyperparameter_optimization']

def identify_model_type(model):
    model_name = model.__class__.__name__.lower()
    if 'vit' in model_name:
        return 'vit'
    elif 'resnet' in model_name:
        return 'resnet'
    elif 'vgg' in model_name:
        return 'vgg'
    elif 'efficientnet' in model_name:
        return 'efficientnet'
    elif 'convnext' in model_name:
        return 'convnext'
    elif 'densenet' in model_name:
        return 'densenet'
    else:
        return 'unknown'
    
def apply_hyperparameters(model, params):
    model_type = identify_model_type(model)
    if model_type == 'vit':
        model.head_drop_rate = params['opt_drop_rate']
        for block in model.blocks:
            block.drop_rate = params['opt_drop_rate']
            block.attn.drop = nn.Dropout(params['attn_drop_rate'])
    elif model_type in ['resnet', 'vgg', 'efficientnet']:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = params['opt_drop_rate']
    elif model_type == 'convnext':
        for block in model.stages:
            for layer in model.stages:
                if hasattr(layer, 'drop_path'):
                    layer.drop_path.drop_prob = params['drop_path_rate']
    elif model_type == 'densenet':
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = params['opt_drop_rate']
        if hasattr(model, 'growth_rate'):
            model.growth_rate = params['growth_rate']
        if hasattr(model, 'compression'):
            model.compression = params['compression_factor']

def get_data_loaders_hyper(config, params):
    train_dataset = CustomDataset(
        root_dir=config['data']['train_dir'],
        info_file=config['data']['train_info_file'],
        augmented_dir=config['data']['augmented_dir'],
        augmented_info_file=config['data']['augmented_info_file'],
        transform=get_transform(config)
    )

    train_size = int((1-config['training']['validation_ratio']) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=params['opt_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['opt_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def objective(trial):
    params = {}

    for param, settings in hp_config['parameters'].items():
        if settings['type'] == 'log_uniform':
            params[param] = trial.suggest_loguniform(param, float(settings['min']), float(settings['max']))
        elif settings['type'] == 'uniform':
            params[param] = trial.suggest_uniform(param, settings['min'], settings['max'])
        elif settings['type'] == 'int':
            params[param] = trial.suggest_int(param, settings['min'], settings['max'])
        elif settings['type'] == 'categorical':
            params[param] = trial.suggest_categorical(param, settings['values'])

    # wandb init
    wandb.init(project="timm-hyperparameter-optimization-ResNet",
               config={
                   "model_name": config['model']['name'],
                   "learning_rate": params['opt_learning_rate'],
                   "batch_size": params['opt_batch_size'],
                   "num_epochs": params['opt_num_epochs'],
                   "drop_rate": params['opt_drop_rate'],
                   "attn_drop_rate": params['attn_drop_rate'],
                   "drop_path_rate": params['drop_path_rate'],
                   "growth_rate": params['growth_rate'],
                   "compression": params['compression_factor']
               }, reinit=True)
    
    # DataLoader
    train_loader, val_loader = get_data_loaders_hyper(config, params)

    # 학습
    device = torch.device(config['training']['device'])
    model = get_model(config).to(device)

    apply_hyperparameters(model, params)

    criterion = get_criterion(config['training']['criterion'])
    optimizer = get_optimizer(config['training']['optimizer'], model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])
    metric_fn = get_metric_function(config['training']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_delta = config['training']['early_stopping']['min_delta']

    for epoch in range(params['opt_num_epochs']):
        train_loss, train_metric, train_class_losses, train_class_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric, val_class_losses, val_class_metric = validate(model, val_loader, criterion, device, metric_fn)

        print(f"Epoch {epoch+1}/{params['opt_num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "val_loss": val_loss,
            "val_metric": val_metric
        })

        # 조기 종료
        if metric_fn.is_better(val_metric, best_val_metric, early_stopping_delta):
            best_val_mertric = val_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    wandb.finish()

    return best_val_mertric

def optimize_hyperparameters():
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=hp_config['n_trials'])
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))
    return trial.params

if __name__ == "__main__":
    best_params = optimize_hyperparameters()