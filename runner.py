
import torch 
from torch import nn

from utils import get_device
from torch.utils.tensorboard import SummaryWriter
from models import CNN, CNNLarger, NeuralNetwork, NeuralNetworkWider, NeuralNetworkDeeper
from data import fashion_mnist_train_test_dataloader

DEFAULT_EPOCHS = 5

CUSTOM_LAYOUT = {
    "loss_accuracy": {
        "loss": ["Multiline", ["Loss/train", "Loss/test"]],
        "accuracy": ["Multiline", ["Accuracy/train", "Accuracy/test"]],
    },
}

class Runner(object):

    def __init__(self):

        self.device = get_device()
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = None

    def train(self, dataloader, epoch):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        self.model.train()
        train_loss, correct = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
    
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            
            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:5d}]')

        train_loss /= num_batches
        correct /= size
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/train', correct, epoch)

    def test(self, dataloader, epoch):
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                test_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f}\n')
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar('Accuracy/test', correct, epoch)

    def run(self):

        batch_size = 32
        train_dataloader, test_dataloader = fashion_mnist_train_test_dataloader(batch_size)
        epoches = 30

        models = [
            NeuralNetwork().to(self.device), 
            NeuralNetworkWider().to(self.device), 
            NeuralNetworkDeeper().to(self.device),
            CNN().to(self.device),
            CNNLarger().to(self.device)
        ]

        for model in models:
            print(model)
            self.model = model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3) # Fixed potimizer

            self.writer = SummaryWriter(log_dir=f'reports/{self.model.__class__.__name__}')
            self.writer.add_custom_scalars(layout=CUSTOM_LAYOUT)
            for epoch in range(1, epoches + 1):
                print(f'Epoch {epoch}\n------------------------------')
                self.train(train_dataloader, epoch)
                self.test(test_dataloader, epoch)

    def run1(self):
        ''' Evaludating CNN with different hyperparameters: batch_size, optimizer, learning_rate, momentum etc.
        '''
        batch_sizes = [1, 8, 16, 32, 64, 128]
        epoches = 30
        for batch_size in batch_sizes:
            print(f'batch_size={batch_size}')
            train_dataloader, test_dataloader = fashion_mnist_train_test_dataloader(batch_size)
            
            
            optimizers = {
                'SGD1': { 'clazz': torch.optim.SGD, 'lr': 1e-3, 'momentum': 0.9 },
                'SGD2': { 'clazz': torch.optim.SGD, 'lr': 1e-4, 'momentum': 0.9 },
                'SGD3': { 'clazz': torch.optim.SGD, 'lr': 1e-3, 'momentum': 0.7 },
                'ADAM1': { 'clazz': torch.optim.Adam, 'lr': 1e-3 },
                'ADAM2': { 'clazz': torch.optim.Adam, 'lr': 1e-4 },
            }
            
            for optimizer_name, optimizer  in optimizers.items():
                # Reset model for every run
                self.model = CNN().to(self.device)
                model_name = self.model.__class__.__name__
                if optimizer_name.startswith('SGD'):
                    self.optimizer = optimizer['clazz'](self.model.parameters(), lr=optimizer['lr'], momentum=optimizer['momentum'])
                else:
                    self.optimizer = optimizer['clazz'](self.model.parameters(), lr=optimizer['lr'])
                
                self.writer = SummaryWriter(log_dir=f'reports/{model_name}-{optimizer_name}-{batch_size}')
                self.writer.add_custom_scalars(layout=CUSTOM_LAYOUT)
                
                for epoch in range(1, epoches + 1):
                    print(f'Epoch {epoch}\n------------------------------')
                    self.train(train_dataloader, epoch)
                    self.test(test_dataloader, epoch)
            