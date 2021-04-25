
import torch_geometric
import torch
from tqdm import tqdm


def train_test_torch_gcn(model, X_torch_train, X_torch_validation, X_torch_test, batch_size=32,
                                learning_rate=0.001,
                                epochs=200,
                                verbose=False,
                                verbose_epochs_accuracy=False):
    def train(model, train_dataloader, criterion, optimizer):
        model.train()

        
        for data in train_dataloader:  # Iterate in batches over the training dataset.

            out = model(data)  # Perform a single forward pass.
          
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.


    def test(model, loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
      
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    tg_train_loader = torch_geometric.data.DataLoader(X_torch_train, batch_size=batch_size, shuffle=True)
    tg_validation_loader = torch_geometric.data.DataLoader(X_torch_validation, batch_size=batch_size, shuffle=True)
    tg_test_loader = torch_geometric.data.DataLoader(X_torch_test, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_acc_epochs = []
    val_acc_epochs = []
    test_acc_epochs = []

    epochs_iterable = list(range(0, epochs ))
    if verbose:
      epochs_iterable = tqdm(epochs_iterable, total=epochs, leave=True, position=0)
    for epoch in epochs_iterable:
        train(model, tg_train_loader, criterion, optimizer)
        train_acc = test(model, tg_train_loader)
        val_acc = test(model, tg_validation_loader)
        test_acc = test(model, tg_test_loader)

        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        test_acc_epochs.append(test_acc)
        if verbose_epochs_accuracy:
          print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return train_acc_epochs, val_acc_epochs, test_acc_epochs

















