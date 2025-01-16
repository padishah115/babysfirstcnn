import torch
import torch.optim as optim

import cnn
from training import train
from validation import validate
from data import load_MNIST



def main():

    train_loader, val_loader = load_MNIST()

    myCNN = cnn.SimpleCNN() #Initialise the CNN using the default number of input channels.
    learning_rate = 1e-2
    optimizer = optim.SGD(params=myCNN.parameters(), lr=learning_rate) #SGD optimizer without any momentum
    n_epochs = 100

    #Perform training of the model over specified number of epochs.
    train(
        model=myCNN, 
        n_epochs=n_epochs, 
        optimizer=optimizer, 
        train_loader=train_loader
        )
    
    accuracy = validate(model=myCNN, val_loader=val_loader)

    print(f"Validation accuracy: {accuracy:.2f}")

    torch.save(myCNN.state_dict(), './model')
    



if __name__ == "__main__": 
    main()