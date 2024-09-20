import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np

from src.config import *
from src.models.model import get_model, get_feature_extractor
from src.data.dataset import get_data_loaders
from src.utils.metrics import get_metric_function

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def calculate_class_loss_metric(y_true, y_pred, criterion, metric_fn):
    classes = np.unique(y_true)
    class_losses = {}
    class_metric = {}

    for cls in classes:
        indices = np.where(y_true == cls)
        size = len(indices[0])
        if size == 0:
            continue

        class_labels = y_true[indices]
        class_preds = y_pred[indices]
        
        class_labels_tensor = torch.tensor(class_labels).to(y_pred.device)
        class_preds_tensor = y_pred[indices]
        loss = criterion(class_preds_tensor, class_labels_tensor).item()
        class_losses[cls] = loss / size

        metric = metric_fn.calculate(class_preds.cpu().numpy(), class_labels)
        class_metric[cls] = metric

    return class_losses, class_metric

def train_one_epoch(model, dataloader, criterion, optimizer, device, metric_fn):
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs  
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_outputs.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    epoch_metric = metric_fn.calculate(all_outputs, all_labels)

    class_losses, class_metric = calculate_class_loss_metric(
        np.array(all_labels), 
        torch.tensor(np.array(all_outputs)).to(device), 
        criterion,
        metric_fn
    )

    return epoch_loss, epoch_metric, class_losses, class_metric

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
        model.head_drop_rate = params['drop_rate']
        for block in model.blocks:
            block.drop_rate = params['drop_rate']
            block.attn.drop = nn.Dropout(params['attn_drop_rate'])
    elif model_type in ['resnet', 'vgg', 'efficientnet']:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = params['drop_rate']
    elif model_type == 'convnext':
        for block in model.stages:
            for layer in model.stages:
                if hasattr(layer, 'drop_path'):
                    layer.drop_path.drop_prob = params['drop_path_rate']
    elif model_type == 'densenet':
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = params['drop_rate']
        if hasattr(model, 'growth_rate'):
            model.growth_rate = params['growth_rate']
        if hasattr(model, 'compression'):
            model.compression = params['compression_factor']

def validate(model, dataloader, criterion, device, metric_fn):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs  
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_outputs.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    epoch_metric = metric_fn.calculate(all_outputs, all_labels)

    class_losses, class_metric = calculate_class_loss_metric(
        np.array(all_labels), 
        torch.tensor(np.array(all_outputs)).to(device), 
        criterion,
        metric_fn
    )

    return epoch_loss, epoch_metric, class_losses, class_metric


def main():
    config = get_config()
    device = torch.device(config['training']['device'])

    model = get_model(config).to(device)
    feature_extractor = get_feature_extractor(config)

    model_type = identify_model_type(model)

    params = {
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_epochs'],
        'drop_rate': config['training']['drop_rate'],
        'attn_drop_rate': config['training']['attn_drop_rate'],
        'drop_rate_path': config['training']['drop_rate_path'],
        'growth_rate': config['training']['growth_rate'],
        'compression_factor': config['training']['compression_factor'],
    }

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config['training']['criterion'])
    optimizer = get_optimizer(config['training']['optimizer'], model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])
    
    metric_fn = get_metric_function(config['training']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['training']['early_stopping']

    apply_hyperparameters(model_type, params)

    # 메인 트레이닝 루프
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_metric, train_class_losses, train_class_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric, val_class_losses, val_class_metric = validate(model, val_loader, criterion, device, metric_fn)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

        #print("Train Class Losses:", train_class_losses)
        #print("Train Class metric:", train_class_metric)
        #print("Val Class Losses:", val_class_losses)
        #print("Val Class metric:", val_class_metric)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if config['training']['lr_scheduler']['monitor'] == 'loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_metric)
        else:
            scheduler.step()

        early_stop_value = val_loss if config['training']['early_stopping']['monitor'] == 'loss' else val_metric
        if metric_fn.is_better(early_stop_value, best_val_metric, early_stopping_config['min_delta']):
            best_val_metric = early_stop_value
            patience_counter = 0
            
            torch.save(model.state_dict(), f"{config['paths']['save_dir']}/best_model1.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    
    #추가 학습 루프
    if config['training']['additional_train']:
        train_loader, val_loader = val_loader, train_loader
        additional_epochs = config['training']['additional_epochs']

        for epoch in range(additional_epochs):
            train_loss, train_metric, train_class_losses, train_class_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
            val_loss, val_metric, val_class_losses, val_class_metric = validate(model, val_loader, criterion, device, metric_fn)

            print(f"Additional Epoch {epoch+1}/{additional_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            
            #print("Train Class Losses:", train_class_losses)
            #print("Train Class metric:", train_class_metric)
            #print("Val Class Losses:", val_class_losses)
            #print("Val Class metric:", val_class_metric)

    torch.save(model.state_dict(), f"{config['paths']['save_dir']}/final_model1.pth")

if __name__ == "__main__":
    main()