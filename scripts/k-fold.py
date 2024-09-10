import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import *
from src.models.model import get_model, get_feature_extractor
from src.data.dataset import get_data_loaders, get_kfold_loaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

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
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs  
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total

def main():
    config = get_config()
    device = torch.device(config['training']['device'])

    for fold, train_loader, val_loader in get_kfold_loaders(config,config['training']['n_splits']):
        print(f"Fold {fold+1}/{config['training']['n_splits']}")
        
        model = get_model(config).to(device)
        criterion = get_criterion(config['training']['criterion'])
        optimizer = get_optimizer(config['training']['optimizer'], model.parameters())
        scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_config = config['training']['early_stopping']

        for epoch in range(config['training']['num_epochs']):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss - early_stopping_config['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{config['paths']['save_dir']}/best_model_fold{fold+1}.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        torch.save(model.state_dict(), f"{config['paths']['save_dir']}/final_model_fold{fold+1}.pth")

if __name__ == "__main__":
    main()