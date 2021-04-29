
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

        cumul_loss = 0
        N = 0
        correct = 0
        for data in train_dataloader:  # Iterate in batches over the training dataset.

            out = model(data)  # Perform a single forward pass.
          
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            cumul_loss += loss.item()
            N += 1

        accuracy = correct / len(train_dataloader.dataset)
        avg_loss = cumul_loss/N 
        return avg_loss, accuracy


    def test(model, loader, criterion):
        model.eval()

        cumul_loss = 0
        N = 0
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
      
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            loss = criterion(out, data.y) 
            cumul_loss += loss.item()
            N += 1

        avg_loss = cumul_loss/N
        accuracy = correct / len(loader.dataset)
        return avg_loss, accuracy  # Derive ratio of correct predictions.

    tg_train_loader = torch_geometric.data.DataLoader(X_torch_train, batch_size=batch_size, shuffle=True)
    tg_validation_loader = torch_geometric.data.DataLoader(X_torch_validation, batch_size=batch_size, shuffle=True)
    tg_test_loader = torch_geometric.data.DataLoader(X_torch_test, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_acc_epochs = []
    val_acc_epochs = []
    test_acc_epochs = []

    train_loss_epochs = []
    val_loss_epochs = []
    test_loss_epochs = []

    epochs_iterable = list(range(0, epochs ))
    if verbose:
      epochs_iterable = tqdm(epochs_iterable, total=epochs, leave=True, position=0)
    for epoch in epochs_iterable:
        avg_train_loss, train_accuracy = train(model, tg_train_loader, criterion, optimizer)

        avg_val_loss, val_accuracy = test(model, tg_validation_loader, criterion)
        avg_test_loss, test_accuracy = test(model, tg_test_loader, criterion)

        train_acc_epochs.append(train_accuracy)
        val_acc_epochs.append(val_accuracy)
        test_acc_epochs.append(test_accuracy)

        train_loss_epochs.append(avg_train_loss)
        val_loss_epochs.append(avg_val_loss)
        test_loss_epochs.append(avg_test_loss)


        if verbose_epochs_accuracy:
          print(f'Epoch: {epoch:03d}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    return train_acc_epochs, val_acc_epochs, test_acc_epochs, train_loss_epochs, val_loss_epochs, test_loss_epochs

















