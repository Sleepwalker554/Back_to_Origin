import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.set_loglevel("warning")

def plot_training_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    title_prefix: Optional[str] = None,
    save_path: Optional[Path] = None
):
    """
    Plot training and validation loss and accuracy curves

    Args:
        epochs: epoch list
        train_loss, val_loss: Training/validation loss
        train_acc, val_acc: Training/validation accuracy (0-1)
        title_prefix: Title prefix (e.g. "Seed 42")
        save_path: Save path (optional)
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
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_seeds_comparison(
    seeds: List[int],
    accuracies: List[float],
    bar_color: str = '#27F5EE',
    mean_color: str = '#B727F5',
    save_path: Optional[Path] = None
):
    """
    Args:
        seeds: seed list
        accuracies: Corresponding accuracy
        bar_color: Bar chart color
        mean_color: Mean line color
        save_path: Save path (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracies]
    bars = ax.bar(range(len(seeds)), acc_pct, color=bar_color, alpha=0.7, edgecolor='black')
    mean_acc = sum(acc_pct) / len(acc_pct)
    ax.axhline(y=mean_acc, color=mean_color, linestyle='--', 
               linewidth=2, label=f'Mean: {mean_acc:.2f}%')
    
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Across Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Label the values on the bars
    for bar, value in zip(bars, acc_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_dataset_comparison(
    dataset_names: List[str],
    accuracies: List[float],
    title: str = "Model Performance Comparison",
    ylabel: str = "Accuracy (%)",
    figsize: Tuple[int, int] = (10, 6),
    ylim: Tuple[float, float] = (0, 100),
    save_path: Optional[Path] = None,
    width: float = 0.5,
    show_values: bool = True,
    value_offset: float = 2.0,
    custom_colors: Optional[dict] = None
):
    """
    Plot bar chart comparing accuracies across different datasets
    
    Args:
        dataset_names: List of dataset names
        accuracies: List of accuracy values (in percentage or 0-1 range)
        title: Chart title
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        ylim: Y-axis limits (min, max)
        save_path: Path to save the figure (optional)
        width: Width of bars
        show_values: Whether to show values on top of bars
        value_offset: Offset for value labels above bars
        custom_colors: Custom color mapping dict, e.g. {'Pitt': '#27F5EE'}
    """
    # Default color mapping for common datasets
    default_color_map = {
        'Pitt': '#27F5EE',
        'ADReSS': '#B727F5', 
        'Lu': '#F5A623'
    }
    
    # Use custom colors if provided, otherwise use defaults
    color_map = custom_colors if custom_colors else default_color_map
    colors = [color_map.get(name, '#888888') for name in dataset_names]
    
    # Convert accuracies to percentage if needed
    acc_pct = [a * 100 if a <= 1.0 else a for a in accuracies]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(dataset_names, acc_pct, color=colors, alpha=0.7, width=width)
    
    # Set labels and title
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    if show_values:
        for bar, acc in zip(bars, acc_pct):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + value_offset,
                    f'{acc:.2f}%',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



