import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(train_loss, test_loss):
    # Check that arrays have same size
    if train_loss.shape[0] != test_loss.shape[0]: 
        print(f"Error: train_loss_hist and test_loss_hist have different sizes!")
    epochs = train_loss.shape[0]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

    ax.plot(np.arange(epochs), train_loss, label="Training", c="red", ls="-", marker="o")
    ax.plot(np.arange(epochs), test_loss, label="Test", c="Navy", ls="-", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(epochs))
    ax.set_ylabel("Loss")
    
    ax.legend()
    ax.grid()

    fig.tight_layout()
    plt.show()