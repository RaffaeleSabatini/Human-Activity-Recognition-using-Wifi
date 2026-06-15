import torch
from torch import nn

def train_loop(model, data_loader, loss_fn, optimizer, device, verbosity, loss_history=None):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    loss = torch.tensor([1e10], device=device)

    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Verbose
        if verbosity and (batch % 10) == 0:
            loss    = loss.item()
            current = batch * batch_size
            print(f"loss: {loss:.5e}  [{current:>5d}/{size:>5d}]")

        # Move data to GPU
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save loss
        if loss_history:
            loss_history.append(loss)



def test_loop(model, dataloader, loss_fn, device, loss_history=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Set the model to evaluation mode
    model.eval()
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Save loss
            if loss_history:
                loss_history.append(test_loss)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
