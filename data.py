import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_MNIST(data_path:str='./data', normalise:bool = True):
    """Function for loading images from the MNIST database.
    
    Args:
        data_path (str): The path at which the data is downloaded.
        normalise (bool): Specifies whether or not the data is normalised.

    Returns:
        training set (DataLoader): The training set returned as a DataLoader
        validation set (DataLoader): The validation set returned as a DataLoader.
    
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

        #Create the transform tensor for normalising images in the database
        normalisation_transform_t = transforms.Normalize(imgs_viewed.mean(dim=1), imgs_viewed.std(dim=1))

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

        return fashionMNIST_train, fashionMNIST_val

    else:

        fashionMNIST_val = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose(transforms.ToTensor())
        )

        return fashionMNIST_unnormalised, fashionMNIST_val