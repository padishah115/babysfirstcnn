import cnn
from loops import train
from loops import validate
from data import load_MNIST
import torch.optim as optim

def main():

    train_loader, val_loader = load_MNIST()

    myCNN = cnn.SimpleCNN()
    learning_rate = 1e-2
    optimizer = optim.SGD(params=myCNN.parameters(), lr=learning_rate) #SGD optimizer without any momentum
    n_epochs = 100

    train(
        model=myCNN, 
        n_epochs=n_epochs, 
        optimizer=optimizer, 
        train_loader=train_loader
        )
    



if __name__ == "__main__": 
    main()