import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from config import LEARNING_RATE, MAX_EPOCHS, WEIGHT_DECAY, XLSR_DIM_HIDDEN, EGEMAPS_DIM_HIDDEN, XLSR_DROPOUT, EGEMAPS_DROPOUT, EGEMAPS_DIM_INPUT, XLSR_DIM_INPUT
from model import AD_XLSR_Model, AD_EGE_Model

def train_one_epoch(model, train_loader, optimizer, device, epoch=None, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create progress bar for batches
    desc = f"Epoch {epoch} - Training" if epoch is not None else "Training"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    
    for batch_data in pbar:
        # Handle both formats: with mask (XLSR) and without mask (eGeMAPS)
        if len(batch_data) == 3:
            features, labels, masks = batch_data
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            logits = model(features, masks)
        else:
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
        
        loss = F.cross_entropy(logits, labels, weight=class_weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
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

    # Create progress bar for validation
    desc = f"Epoch {epoch} - Validation" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_data in pbar:
            # Handle both formats: with mask (XLSR) and without mask (eGeMAPS)
            if len(batch_data) == 3:
                features, labels, masks = batch_data
                features = features.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                logits = model(features, masks)
            else:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
            
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            predictions = torch.argmax(logits, dim=1)

            # Overall statistics
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
            
            # Update progress bar
            current_loss = total_loss / (pbar.n + 1)
            current_acc = correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    # Per-class accuracy
    control_acc = control_correct / control_total if control_total > 0 else 0
    dementia_acc = dementia_correct / dementia_total if dementia_total > 0 else 0

    # F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, control_acc, dementia_acc, f1_score


def train(seed, train_loader, val_loader, output_dir, device, xlsr=True, class_weight_control=1.0, class_weight_dementia=1.0):
    """
    Training pipeline

    Args:
        seed: Random seed
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save models
        device: Device to train on (cpu/cuda/mps)
        xlsr: Whether using XLSR features (True) or eGeMAPS features (False)
        class_weight_control: Weight for Control class (0) in loss function
        class_weight_dementia: Weight for Dementia class (1) in loss function

    Returns:
        seed: The seed used
        best_metrics: Dictionary of best validation metrics
        training_history: Dictionary of training history (epochs, losses, accuracies)
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create class weights tensor
    class_weights = torch.tensor([class_weight_control, class_weight_dementia], 
                                  dtype=torch.float32, device=device)

    # Create save directory
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Create model for xlsr or egemaps features
    if xlsr:
        model = AD_XLSR_Model(dropout=XLSR_DROPOUT).to(device)
    else:
        model = AD_EGE_Model(dim_input=25,
                             dim_hidden=14,
                             dropout=0.2).to(device) 

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_list = []

    # Early stopping
    best_val_acc = 0
    best_metrics = {}
    patience = 10
    patience_counter = 0
    best_epoch = 0
    stopped_epoch = 0

    # Training loop
    for epoch in range(MAX_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch=epoch+1, class_weights=class_weights)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, control_acc, dementia_acc, f1 = validate(model, val_loader, device, epoch=epoch+1, class_weights=class_weights)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epochs_list.append(epoch)

        # Save best model
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

        # Early stopping check
        if patience_counter >= patience:
            stopped_epoch = epoch + 1
            break

    # Print final results
    print(f"Seed {seed}:\n"
          f"Val Acc={best_metrics['val_acc']*100:.2f}%, "
          f"F1={best_metrics['f1_score']:.4f}, "
          f"Val Loss={best_metrics['val_loss']:.4f}\n"
          f"Control Acc={best_metrics['control_acc']*100:.2f}%, "
          f"Dementia Acc={best_metrics['dementia_acc']*100:.2f}%\n"
          f"Early stop at: epoch {stopped_epoch if stopped_epoch > 0 else MAX_EPOCHS} (best epoch: {best_epoch})")

    # Return training history along with best metrics
    training_history = {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }

    return seed, best_metrics, training_history
