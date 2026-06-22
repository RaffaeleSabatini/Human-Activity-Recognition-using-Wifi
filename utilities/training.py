import torch
import numpy as np
from torch import nn
from time import time

def train_loop(model, data_loader, loss_fn, optimizer, device, verbosity):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    batch_size = data_loader.batch_size
    avg_loss = 0
    start = time()

    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Move data to GPU
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        logits = model(X, return_pred=False)
        loss            = loss_fn(logits, y)
        avg_loss       += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Verbose
        if verbosity and (batch % 10) == 0:
            loss    = loss.item()
            current = (batch+1) * batch_size
            print(f"Loss: {loss:.5e} --- Samples: {current:>5d}/{size:>5d}")
    
    avg_loss /= num_batches
    return avg_loss, time()-start

def test_loop(model, dataloader, loss_fn, device, verbosity):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Set the model to evaluation mode
    model.eval()
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            logits = model(X, return_pred=False)  # Shape: (batch, activity)
            test_loss += loss_fn(logits, y).item()

            if torch.any(y != y[0]):
                print(f"Error: in batch {i} antennas produced different labels for the same task!")
                break
            
            # Soft-fusion
            probabilities     = torch.softmax(logits, dim=1)
            avg_probabilities = torch.mean(probabilities, dim=0)
            pred = torch.argmax(avg_probabilities)

            if y[0].item() == pred.item():
                correct += 1

    test_loss /= num_batches
    accuracy = correct/num_batches

    if verbosity:
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, accuracy*100

#---------------------------------------------------------------------------------------------------

def train_model(model, train_data_loader, test_data_loader, epochs, loss_fn, optimizer, device, verbosity):
    model.to(device)
    train_loss_history = np.zeros(epochs)
    test_loss_history  = np.zeros(epochs)
    train_time_history = np.zeros(epochs)
    start = time()

    print("Starting model training...")
    print(f"Completed epochs: 0/{epochs} ------ Time (total): {0:.2f} ------ Time (relative): {0:.2f}\n")
    for epoch in range(epochs):
        relative_start = time()

        train_loss, train_time = train_loop(model, train_data_loader, loss_fn, optimizer, device, verbosity)
        test_loss, accuracy    = test_loop(model, test_data_loader, loss_fn, device, verbosity)

        train_loss_history[epoch] = train_loss
        test_loss_history[epoch]  = test_loss
        train_time_history[epoch] = train_time

        print(f"Completed epochs: {epoch+1}/{epochs} ------ Time (total): {time()-start:.2f} ------ Time (relative): {time()-relative_start:.2f}\n")
    
    return train_loss_history, test_loss_history, train_time_history