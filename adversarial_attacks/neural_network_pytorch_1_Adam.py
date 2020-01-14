import numpy as np
from sklearn.preprocessing import OneHotEncoder
from IPython.core.debugger import set_trace
"""
input:natural image arr:n*w*h*c
return: quantisized image n*w*h*c
"""
def quantization(arr,k):
    quant = np.zeros(arr.shape)
    for i in range(1,k):
        quant[arr>1.0*i/k]+=1
    return quant

"""
input:quantisized img shape:n*w*h*c
retun:one-hot coded image shape:n*w*h*c*k
"""
f = open("output.txt","a")
def onehot(arr,k):
    n,w,h = arr.shape
    arr = arr.reshape(n,-1)
    enc=OneHotEncoder(sparse=False)
    arr = enc.fit_transform(arr)
    arr = arr.reshape(n,w,h,k)
    arr = arr.transpose(0,3,1,2)
    return arr

"""
input:one-hot coded img shape:n*w*h*c*k
retun:trmp coded image shape:n*w*h*c*k
"""
def tempcode(arr,k):
    tempcode = np.zeros(arr.shape)
    for i in range(k):
        tempcode[:,i,:,:] = np.sum(arr[:,:i+1,:,:],axis=1)
    return tempcode
    
"""
from a thermometerencoding image to a mormally coded image, for some visulization usage
"""
def temp2img(tempimg,k):
    img = np.sum(tempimg,axis=1)
    img = np.ones(img.shape)*(k+1)-img
    img = img*1.0/k
    return img

def getMask(x,epsilon,k):
    n,w,h = x.shape
    mask = np.zeros((n,k,w,h))
    low = x - epsilon
    low[low < 0] = 0
    high = x + epsilon
    high[high > 1] = 1
    for i in range(k+1):
        interimg = (i*1./k)*low + (1-i*1./k)*high
        mask+=onehot(quantization(interimg,k),k)
    mask[mask>1] = 1
    return mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(15, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model = Net()
model = model.cuda()

def LSPGDonechannel(data,target,epsilon,k,delta,xi,step,criterion):
    datanumpy = data.numpy()
    data0 = datanumpy[:,0,:,:]
    mask = getMask(data0,epsilon,k)
    u = np.random.random(mask.shape)-(1-mask)*1e10
    T = 1.0
    u = Variable(torch.Tensor(u).cuda(),requires_grad=True)
    z = F.softmax(u/T,dim=1)
    z = torch.cumsum(z,dim=1)
    for t in range(step):
        out = model(z)
        loss = criterion(out,target)
        if u.grad!=None:
            u.grad.data._zero()
        loss.backward()
        grad = u.grad
        u = xi*torch.sign(grad) + u
        u = Variable(u.data,requires_grad=True)
        z = F.softmax(u/T,dim=1)
        z = torch.cumsum(z,dim=1)
        T = T*delta
    attackimg = np.argmax(u.data.cpu().numpy(),axis=1)
    themattackimg = tempcode(onehot(attackimg,k),k)
    return themattackimg

criterion = nn.CrossEntropyLoss()
level=15
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def trainnat(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = Variable(target)
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(onehot(quantization(data,level),level)))
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train (normal set) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            f.write('Train (normal set) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = Variable(target)
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(onehot(quantization(data,level),level)))
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest (true set) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f.write('\nTest (true set) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def trainadv(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = Variable(target.cuda())
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(tempcode(onehot(quantization(data,level),level),level)))
        data = Variable(torch.Tensor(data).cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train (Temp) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            f.write('Train (Temp) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    
def testadv():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.cuda()
        target = Variable(target)
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(tempcode(onehot(quantization(data,level),level),level)))
        data = Variable(torch.Tensor(data).cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest (temp) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f.write('\nTest (temp) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def our_model(epoch):
    i=0
    for param in model.parameters():
        if(i!=0):
            param.requires_grad = False
        i+=1

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = Variable(target.cuda())
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(tempcode(onehot(quantization(data,level),level),level)))
        data = Variable(torch.Tensor(data).cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            f.write('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def our_modelPGD(epoch):
    i=0
    for param in model.parameters():
        if(i!=0):
	        param.requires_grad = False
	    
        i+=1

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = Variable(target.cuda())
        data = LSPGDonechannel(data=data,target=target,epsilon=0.3,k=level,delta=1.2,xi=1.0,step=2,criterion=criterion)
        data = Variable(torch.Tensor(data).cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            f.write('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test_PGD():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.cuda()
        data = LSPGDonechannel(data=data,target=target,epsilon=0.3,k=level,delta=1.2,xi=1.0,step=2,criterion=criterion)
        data = Variable(torch.Tensor(data).cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest (AutoEncoder on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f.write('\nTest (AutoEncoder on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))





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

AutoEncoderModel = AutoEncoder()
AutoEncoderModel = AutoEncoderModel.cuda()
optimizer = optim.Adam(AutoEncoderModel.parameters(), lr=1e-4)
distance = nn.MSELoss()
num_epochs = 1


def train_autoencoder():
	AutoEncoderModel.train()
	for epoch in range(num_epochs):
	    for batch_idx, (data, target) in enumerate(train_loader):
	        target = Variable(target.cuda())
	        data1 = LSPGDonechannel(data=data,target=target,epsilon=0.3,k=level,delta=1.2,xi=1.0,step=2,criterion=criterion)
	        data1 = Variable(torch.Tensor(data1).cuda())
	        data = data.numpy()[:,0,:,:]
	        data = Variable(torch.Tensor(onehot(quantization(data,level),level))).cuda()
	        optimizer.zero_grad()
	        output = AutoEncoderModel(data1)
	        loss = distance(output, data)
	        loss.backward()
	        optimizer.step()
	        if batch_idx % 10 == 0:
	            print('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                epoch, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.data.item()))
	            f.write('Train (Our_model) Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
	                epoch, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.data.item()))



def test_Autoencoder():
    AutoEncoderModel.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        dact = Variable(data).cuda()
        target = Variable(target.cuda())
        data1 = LSPGDonechannel(data=data,target=target,epsilon=0.3,k=level,delta=1.2,xi=1.0,step=2,criterion=criterion)
        data1 = Variable(torch.Tensor(data1).cuda())
        data = data.numpy()[:,0,:,:]
        data = Variable(torch.Tensor(onehot(quantization(data,level),level))).cuda()
        output = AutoEncoderModel(data1)
        test_loss += distance(output, data).data.item() 
        pred = torch.tensor(output.data.max(1, keepdim=True)[1],dtype=torch.float).cuda()
        correct += pred.eq(dact.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest (Auto Encode on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f.write('\nTest (Auto Encoder on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

trainnat(0)
# test()
# testadv()
# our_model(0)
# testadv()
# test()
# test_PGD()
# our_modelPGD(2)
# test()
# testadv()
# test_PGD()
train_autoencoder()
new_model = nn.Sequential(AutoEncoderModel,model)
def test_new_model():
    new_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.cuda()
        data = LSPGDonechannel(data=data,target=target,epsilon=0.3,k=level,delta=1.2,xi=1.0,step=2,criterion=criterion)
        data = Variable(torch.Tensor(data).cuda())
        output = new_model(data)
        test_loss += criterion(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest (AutoEncoder on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f.write('\nTest (AutoEncoder on PGD) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# test_Autoencoder()
test_new_model()
f.close()
