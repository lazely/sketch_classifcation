import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import KFold
from src.config import *
from src.models.model import get_model, get_feature_extractor
from src.data.dataset import get_data_loaders, get_full_dataset
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch['pixel_values'].to(device), batch['label'].to(device)

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
            inputs, labels = batch['pixel_values'].to(device), batch['label'].to(device)

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
    
    if config['training'].get('use_kfold', False):
        full_dataset = get_full_dataset(config)
        n_splits = config['training'].get('k_fold', 5)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        best_overall_val_loss = float('inf')
        best_fold = -1

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
            print(f"\nTraining fold {fold + 1}/{n_splits}")
            
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

            # Initialize a new model for each fold
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

                # Learning rate scheduler step
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Early stopping check
                if val_loss < best_val_loss - early_stopping_config['min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model for this fold
                    torch.save(model.state_dict(), f"{config['paths']['save_dir']}/best_model_fold_{fold+1}.pth")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_config['patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            fold_results.append((fold + 1, best_val_loss))
            
            # Update best overall model if this fold performed better
            if best_val_loss < best_overall_val_loss:
                best_overall_val_loss = best_val_loss
                best_fold = fold + 1

        # Print results for all folds
        print("\nResults for all folds:")
        for fold, val_loss in fold_results:
            print(f"Fold {fold}: Validation Loss = {val_loss:.4f}")

        print(f"\nBest performing fold: {best_fold} with validation loss: {best_overall_val_loss:.4f}")

        # Copy the best model to the final model
        best_model_path = f"{config['paths']['save_dir']}/best_model_fold_{best_fold}.pth"
        final_model_path = f"{config['paths']['save_dir']}/final_model.pth"
        torch.save(torch.load(best_model_path), final_model_path)

        print(f"Best model saved as final model: {final_model_path}")

    else:
        # Code for non-k-fold training
        train_loader, val_loader = get_data_loaders(config)
        # ... (rest of the non-k-fold training code)

if __name__ == "__main__":
    main()