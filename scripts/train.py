import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import get_config,get_criterion,get_optimizer
from src.models.model import get_model, get_feature_extractor
from src.data.dataset import get_data_loaders

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

    model = get_model(config).to(device)
    feature_extractor = get_feature_extractor(config)

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config['training']['criterion'])
    optimizer = get_optimizer(config['training']['optimizer'], model.parameters())

    for epoch in range(config['training']['num_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), f"{config['paths']['save_dir']}/final_model.pth")

if __name__ == "__main__":
    main()