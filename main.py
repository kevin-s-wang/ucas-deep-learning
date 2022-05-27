from sklearn.utils import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils import get_device
from models import CNN1, NeuralNetwork, CNN


# Hosting datasets leveraging a local web server
# git clone https://github.com/zalandoresearch/fashion-mnist.git
# docker run -d -p 80:80 -v $(pwd):/usr/share/nginx/html nginx
device = get_device()


def prepare_data(batch_size):
    datasets.FashionMNIST.mirrors = [
        'http://localhost/'
    ]

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:5d}]')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f}\n')



def main():
    # writer = SummaryWriter()

    train_dataloader, test_dataloader = prepare_data(32)

    for X, y in test_dataloader:
        print(f'Shape of X [N, C, H, W]: {X.shape}')
        print(f'Shape of y: {y.shape} {y.dtype}')
        break
    
    print(f'Using {device} device')
        
    #model = NeuralNetwork().to(device)
    model = CNN(1).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    epochs = 20
    for t in range(epochs):
        print(f'Epoch {t+1}\n------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    # writer.flush()
    # writer.close()


if __name__ == '__main__':
    main()