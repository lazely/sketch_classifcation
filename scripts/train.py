import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.models.model import get_model, get_feature_extractor
from src.data.dataset import get_data_loaders
from src.utils.metrics import get_metric_function

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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
    return epoch_loss, epoch_metric

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
    return epoch_loss, epoch_metric

def main():
    config = get_config()
    device = torch.device(config['training']['device'])

    model = get_model(config).to(device)
    feature_extractor = get_feature_extractor(config)

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config['training']['criterion'])
    optimizer = get_optimizer(config['training']['optimizer'], model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])
    
    metric_fn = get_metric_function(config['training']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['training']['early_stopping']

    for epoch in range(config['training']['num_epochs']):
        train_loss, train_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric = validate(model, val_loader, criterion, device, metric_fn)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

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
            
            torch.save(model.state_dict(), f"{config['paths']['save_dir']}/best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    torch.save(model.state_dict(), f"{config['paths']['save_dir']}/final_model.pth")

if __name__ == "__main__":
    main()