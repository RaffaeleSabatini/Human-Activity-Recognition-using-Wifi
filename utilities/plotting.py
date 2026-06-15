import matplotlib.pyplot as plt
import torch

def plot_loss(train_loss, test_loss):
    # Check that arrays have same size
    if len(train_loss) != len(test_loss): 
        print(f"Error: train_loss_hist and test_loss_hist have different sizes! {}")