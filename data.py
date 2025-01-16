import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch.nn as nn

def load_MNIST(data_path:str='./data', normalise:bool = True):
    """Function for loading images from the MNIST database.
    
    Args:
        data_path (str): The path at which the data is downloaded.
        normalise (bool): Specifies whether or not the data is normalised.

    Returns:
        train_loader (DataLoader): The training set returned as a DataLoader
        val_loader (DataLoader): The validation set returned as a DataLoader.
    
    """

    #Load the FashionMNIST database without normalisation
    fashionMNIST_unnormalised = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    if normalise:
        imgs = torch.stack([img_t for img_t, label in fashionMNIST_unnormalised], dim=3) #stack the images along a new dimension
        imgs_viewed = imgs.view(1, -1)

        fashionMNIST_train = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(imgs_viewed.mean(dim=1), imgs_viewed.std(dim=1))
                ])
        )

        fashionMNIST_val = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(imgs_viewed.mean(dim=1), imgs_viewed.std(dim=1))
                ])
        )

        train_loader = DataLoader(dataset=fashionMNIST_train, batch_size=64, shuffle=False)
        val_loader = DataLoader(dataset=fashionMNIST_val, batch_size=64, shuffle=False)

    else:

        fashionMNIST_val = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose(transforms.ToTensor())
        )

        train_loader = DataLoader(dataset=fashionMNIST_unnormalised, batch_size=64, shuffle=False)
        val_loader = DataLoader(dataset=fashionMNIST_val, batch_size=64, shuffle=False)


    
    return train_loader, val_loader
    

# train_set, val_set = load_MNIST()

# img_t, label = train_set[0]

# conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

# output = conv(img_t)