import torch
import tqdm


def train(model, dataloader, criterion, optimizer, num_epochs, device='cpu'):
    model.to(device)
    loss_history = []

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training", leave=False):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)


    return loss_history


    