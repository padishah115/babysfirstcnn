import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def validate(model:nn.Module, val_loader:DataLoader):
    """Tests model performance after training against a validation set.
    
    Args:
        model (nn.Module): The pre-trained model whose performance is to be tested.
        val_loader (DataLoader): The validation set, wrapped in a DataLoader, for testing performance.

    Returns:
        accuracy (float): Percentage of classifications by the model which were correct.
    
    """

    correct = 0
    total = 0

    #Loops through the images in the validation loader.
    for imgs, labels in val_loader:
        outputs = model(imgs)
        predicted_vals, predicted_labels = torch.max(outputs, dim=1)

        total += imgs.shape[0] #Number of images
        correct += int((labels==predicted_labels).sum()) #number of correct classifications.

    accuracy = correct / total

    return accuracy