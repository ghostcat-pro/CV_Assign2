"""
Training visualization utilities for plotting loss and metrics during training.
"""
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional


class TrainingPlotter:
    """Track and plot training/validation metrics during training."""
    
    def __init__(self, save_dir: str = "logs", plot_name: str = "training_plot"):
        """
        Initialize the plotter.
        
        Args:
            save_dir: Directory to save plots
            plot_name: Base name for saved plot files
        """
        self.save_dir = save_dir
        self.plot_name = plot_name
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'epochs': []
        }
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_iou: float, val_iou: float):
        """
        Update history with new epoch metrics.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_iou: Training mIoU
            val_iou: Validation mIoU
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_iou'].append(train_iou)
        self.history['val_iou'].append(val_iou)
    
    def plot(self, show: bool = False, save: bool = True):
        """
        Generate and save training plots.
        
        Args:
            show: Whether to display the plot (default: False for non-interactive)
            save: Whether to save the plot to disk
        """
        if not self.history['epochs']:
            print("No training history to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = self.history['epochs']
        
        # Plot 1: Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: mIoU
        ax2.plot(epochs, self.history['train_iou'], 'b-', label='Train mIoU', linewidth=2)
        ax2.plot(epochs, self.history['val_iou'], 'r-', label='Val mIoU', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('mIoU', fontsize=12)
        ax2.set_title('Training and Validation mIoU', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f"{self.plot_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_history(self):
        """Save training history to a text file."""
        history_path = os.path.join(self.save_dir, f"{self.plot_name}_history.txt")
        
        with open(history_path, 'w') as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_mIoU\tVal_mIoU\n")
            for i in range(len(self.history['epochs'])):
                f.write(f"{self.history['epochs'][i]}\t"
                       f"{self.history['train_loss'][i]:.6f}\t"
                       f"{self.history['val_loss'][i]:.6f}\t"
                       f"{self.history['train_iou'][i]:.6f}\t"
                       f"{self.history['val_iou'][i]:.6f}\n")
        
        print(f"Training history saved to: {history_path}")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics achieved during training.
        
        Returns:
            Dictionary with best metrics
        """
        if not self.history['epochs']:
            return {}
        
        best_val_iou_idx = self.history['val_iou'].index(max(self.history['val_iou']))
        min_val_loss_idx = self.history['val_loss'].index(min(self.history['val_loss']))
        
        return {
            'best_val_iou': max(self.history['val_iou']),
            'best_val_iou_epoch': self.history['epochs'][best_val_iou_idx],
            'min_val_loss': min(self.history['val_loss']),
            'min_val_loss_epoch': self.history['epochs'][min_val_loss_idx],
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_iou': self.history['train_iou'][-1],
            'final_val_iou': self.history['val_iou'][-1]
        }


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                         train_ious: List[float], val_ious: List[float],
                         save_path: Optional[str] = None, show: bool = False):
    """
    Quick utility to plot training curves from lists.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_ious: List of training mIoU per epoch
        val_ious: List of validation mIoU per epoch
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: mIoU
    ax2.plot(epochs, train_ious, 'b-', label='Train mIoU', linewidth=2)
    ax2.plot(epochs, val_ious, 'r-', label='Val mIoU', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mIoU', fontsize=12)
    ax2.set_title('Training and Validation mIoU', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
