import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class AutoEncoder(nn.Module):
        def __init__(self):
                super(AutoEncoder,self).__init__()
                self.encoder = nn.Sequential(
                        nn.Conv2d(15,32,kernel_size=5),
                        nn.Sigmoid(),
                        nn.Conv2d(32,64,kernel_size=5),
                        nn.Sigmoid())
                self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(64,32,kernel_size=5),
                        nn.Sigmoid(),
                        nn.ConvTranspose2d(32,15,kernel_size=5),
                        nn.Sigmoid())

        def forward(self,x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        def data_loader(self):
            train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
                    batch_size=100, shuffle=True)

            return train_loader.dataset.data
