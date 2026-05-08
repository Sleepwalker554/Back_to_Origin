import matplotlib.pyplot as plt
from typing import List, Optional
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib
matplotlib.set_loglevel("warning")

def plot_training_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    title_prefix: Optional[str] = None,
):
    """
    Plot training and validation loss and accuracy curves

    Args:
        epochs: epoch list
        train_loss, val_loss: Training/validation loss
        train_acc, val_acc: Training/validation accuracy (0-1)
        title_prefix: Title prefix (e.g. "Seed 42")
    """
    # Define colors inside the function
    train_color = '#2E86AB'
    val_color = '#A23B72'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    title_base = f"{title_prefix}: " if title_prefix else ""

    # Loss curve
    ax1.plot(epochs, train_loss, 'o-', label='Train', color=train_color, linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 's-', label='Val', color=val_color, linewidth=2, markersize=4)
    ax1.set_title(f'{title_base}Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    train_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in train_acc]
    val_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in val_acc]
    
    ax2.plot(epochs, train_acc_pct, 'o-', label='Train', color=train_color, linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc_pct, 's-', label='Val', color=val_color, linewidth=2, markersize=4)
    ax2.set_title(f'{title_base}Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()