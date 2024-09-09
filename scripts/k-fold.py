import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import KFold
from src.config import get_config
from src.models import get_model
from src.data import get_data_loaders, get_feature_extractor
from src.utils import get_criterion, get_optimizer, get_lr_scheduler
import numpy as np
import torch
import tqdm

def train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} - Training"):
            inputs, labels = batch['pixel_values'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, labels = batch['pixel_values'].to(device), batch['label'].to(device)

                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f"{config['paths']['save_dir']}/best_model_fold.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    return best_val_loss

def kfold_cross_validation(config, n_splits=5):
    device = torch.device(config['training']['device'])
    full_dataset = get_full_dataset(config)  # 전체 데이터셋을 로드하는 함수 필요
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Training fold {fold + 1}/{n_splits}")
        
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=config['training']['batch_size'])
        
        model = get_model(config).to(device)
        criterion = get_criterion(config['training']['criterion'])
        optimizer = get_optimizer(config['training']['optimizer'], model.parameters())
        scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])
        
        best_val_loss = train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)
        fold_scores.append(best_val_loss)
        
    return np.mean(fold_scores), np.std(fold_scores)

def main():
    config = get_config()
    device = torch.device(config['training']['device'])

    feature_extractor = get_feature_extractor(config)
    
    # K-fold 설정
    n_splits = config['training'].get('k_fold', 5)  # config에서 k_fold 값을 가져오거나 기본값 5 사용
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 전체 데이터셋 로드
    full_dataset = get_full_dataset(config)  # 이 함수는 구현되어 있다고 가정

    best_overall_val_loss = float('inf')
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold+1}/{n_splits}")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=config['training']['batch_size'])

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

        # 각 fold의 best validation loss 비교
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_fold = fold + 1

    print(f"Best performing fold: {best_fold} with validation loss: {best_overall_val_loss:.4f}")

    # 최고 성능의 모델을 최종 모델로 복사
    best_model_path = f"{config['paths']['save_dir']}/best_model_fold_{best_fold}.pth"
    final_model_path = f"{config['paths']['save_dir']}/final_model.pth"
    torch.save(torch.load(best_model_path), final_model_path)

    print(f"Best model saved as final model: {final_model_path}")

if __name__ == "__main__":
    main()