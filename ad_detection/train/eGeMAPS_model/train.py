import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from .config import LEARNING_RATE, MAX_EPOCHS, WEIGHT_DECAY, DROPOUT, ETA_MIN, PATIENCE
from .model import AD_EGE_Model

def train_one_epoch(model, train_loader, optimizer, device, epoch=None, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    desc = f"Epoch {epoch} - Training" if epoch is not None else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)

    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)

        loss = F.cross_entropy(logits, labels, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        current_loss = total_loss / (pbar.n + 1)
        current_acc = correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, device, epoch=None, class_weights=None):
    """Validate model and compute detailed metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # For per-class metrics (0: Control, 1: Dementia)
    control_correct = 0
    control_total = 0
    dementia_correct = 0
    dementia_total = 0

    # For F1 score
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    desc = f"Epoch {epoch} - Validation" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    with torch.no_grad():
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)

            loss = F.cross_entropy(logits, labels, weight=class_weights)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for pred, label in zip(predictions, labels):
                if label == 0:  # Control
                    control_total += 1
                    if pred == label:
                        control_correct += 1
                else:  # Dementia
                    dementia_total += 1
                    if pred == label:
                        dementia_correct += 1

                # F1 score components (Dementia as positive class)
                if pred == 1 and label == 1:
                    true_positives += 1
                elif pred == 1 and label == 0:
                    false_positives += 1
                elif pred == 0 and label == 1:
                    false_negatives += 1

            current_loss = total_loss / (pbar.n + 1)
            current_acc = correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    control_acc = control_correct / control_total if control_total > 0 else 0
    dementia_acc = dementia_correct / dementia_total if dementia_total > 0 else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, control_acc, dementia_acc, f1_score


def train(seed, train_loader, val_loader, output_dir, device, class_weight_control=1.0, class_weight_dementia=1.0):
    """
    Training pipeline for eGeMAPS features.

    Args:
        seed: Random seed
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save models
        device: Device to train on (cpu/cuda/mps)
        class_weight_control: Weight for Control class (0) in loss function
        class_weight_dementia: Weight for Dementia class (1) in loss function

    Returns:
        seed: The seed used
        best_metrics: Dictionary of best validation metrics
        training_history: Dictionary of training history (epochs, losses, accuracies)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    class_weights = torch.tensor([class_weight_control, class_weight_dementia],
                                  dtype=torch.float32, device=device)

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = AD_EGE_Model(dropout=DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_list = []

    best_val_acc = 0
    best_metrics = {}
    patience_counter = 0
    best_epoch = 0
    stopped_epoch = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch+1, class_weights=class_weights)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        scheduler.step()

        val_loss, val_acc, control_acc, dementia_acc, f1 = validate(model, val_loader, device, epoch=epoch+1, class_weights=class_weights)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epochs_list.append(epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_metrics = {
                'val_acc': val_acc,
                'val_loss': val_loss,
                'control_acc': control_acc,
                'dementia_acc': dementia_acc,
                'f1_score': f1
            }
            patience_counter = 0
            torch.save(model.state_dict(), seed_dir / 'best.pth')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            stopped_epoch = epoch + 1
            break

    print(f"Seed {seed}:\n"
          f"Val Acc={best_metrics['val_acc']*100:.2f}%, "
          f"F1={best_metrics['f1_score']:.4f}, "
          f"Val Loss={best_metrics['val_loss']:.4f}\n"
          f"Control Acc={best_metrics['control_acc']*100:.2f}%, "
          f"Dementia Acc={best_metrics['dementia_acc']*100:.2f}%\n"
          f"Early stop at: epoch {stopped_epoch if stopped_epoch > 0 else MAX_EPOCHS} (best epoch: {best_epoch})")

    training_history = {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

    return seed, best_metrics, training_history
