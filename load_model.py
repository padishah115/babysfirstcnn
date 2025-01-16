import cnn
import torch

def main():
    loaded_model = cnn.SimpleCNN()
    loaded_model.load_state_dict(torch.load('model.pt'))

    numel_list = [p.numel() for p in loaded_model.parameters() if p.requires_grad == True]
    print(f'Parameters list: {numel_list}', f'Total: {sum(numel_list)}')


if __name__ == "__main__":
    main()

