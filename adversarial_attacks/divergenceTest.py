from Autoencoder import AutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from thermometerEncoder import ThermometerEncoder 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=100, shuffle=True)
dat = train_loader.dataset[0][0].cpu().numpy()
Tfit = ThermometerEncoder(dat, 15)
Z = Tfit.quantization()
oneEnc = Tfit.OneHotEncode()
print(Z)
print(oneEnc)
NormalAutoencoder= AutoEncoder()
AdversarialAutoencoder = AutoEncoder()
