import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

def train(model:nn.Module, 
          n_epochs:int, 
          optimizer:optim.Optimizer, 
          train_loader:DataLoader):
    """Training loop for the convolutional neural network.
    
    Args:
        model (nn.Module): The neural network to be passed for training
        n_epochs (int): The number of training epochs
        optimizer (optim.Optimizer): The preferred choice of optimizer for training the neural network
        train_loader (DataLoader): DataLoader containing the training data

    Returns:
        None
    
    """
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0 #initialise the loss

        for imgs, labels in train_loader:

            outputs = model(imgs)
            loss = model.loss_fn(outputs, labels)

            optimizer.zero_grad() #Zeros the gradients to prevent accumulation over many optimization steps
            loss.backward()
            optimizer.step() #perform optimization step

            loss_train += loss

        if epoch == 1 or epoch % 10 == 0:
            print(f"Loss at epoch {epoch}: {loss_train/len(train_loader)}") #returns the loss, normalised to the number of data



# def validate(model:nn.Module, val_loader:DataLoader):
#     """Tests model performance after training against a validation set.
    
#     Args:
#         model (nn.Module): The pre-trained model whose performance is to be tested.
#         val_loader (DataLoader): The validation set, wrapped in a DataLoader, for testing performance.
    
#     """

#     for imgs, labels in val