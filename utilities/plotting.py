import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np



def plot_dataset(cols, rows, dataset, activities, labels):
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()  

    for i in range(cols * rows):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label_idx = dataset[sample_idx]
        
        ax = axes[i]
        im = ax.imshow(img[0, :, :], aspect='auto', cmap='viridis')
        
        ax.set_title(f"{activities[label_idx]} ({labels[label_idx]})", fontsize=10, pad=8)
        
        ax.set_xlabel("Time (bin)", fontsize=8)
        ax.set_ylabel(r"$v_p \cos \alpha$ (bin)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Power (db)', fontsize=8)

    plt.tight_layout()
    plt.show()



def plot_loss(train_loss, test_loss, train_accuracy, test_accuracy):
    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --------------------------------------------------------------------------
    # GRAPH 1: LOSS HISTORY
    # --------------------------------------------------------------------------
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, test_loss, 'r-s', label='Test Loss (Fusion)', linewidth=2, markersize=4)
    
    min_test_loss_idx = np.argmin(test_loss)
    ax1.axvline(x=min_test_loss_idx + 1, color='gray', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({min_test_loss_idx + 1})')
    ax1.plot(min_test_loss_idx + 1, test_loss[min_test_loss_idx], 'go', markersize=10, 
             label=f'Min Test Loss: {test_loss[min_test_loss_idx]:.4f}')
    
    ax1.set_title('Andamento della Loss durante l\'Addestramento', fontsize=12, pad=10)
    ax1.set_xlabel('Epoca', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')
    
    # --------------------------------------------------------------------------
    # GRAPH 2: ACCURACY HISTORY
    # --------------------------------------------------------------------------
    ax2.plot(epochs, train_accuracy, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, test_accuracy, 'r-s', label='Test Accuracy (Fusion)', linewidth=2, markersize=4)
    
    # Evidenzia il punto di massima accuratezza nel Test
    max_test_acc_idx = np.argmax(test_accuracy)
    ax2.plot(max_test_acc_idx + 1, test_accuracy[max_test_acc_idx], 'go', markersize=10, 
             label=f'Max Test Accuracy: {test_accuracy[max_test_acc_idx]:.1f}%')
    
    ax2.set_title('Andamento dell\'Accuratezza', fontsize=12, pad=10)
    ax2.set_xlabel('Epoca', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.show()