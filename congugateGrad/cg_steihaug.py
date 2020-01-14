import numpy as np
from absl import flags
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import autograd
writer = SummaryWriter()

FLAGS = flags.FLAGS
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def cg_steihaug(loss, inputs, net, epsilon =0.001):
	grad_val = autograd.grad(loss, inputs, create_graph=True)
	z = grad_val[0] @ inputs
	z.backward()

	return inputs.grad


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # x_fgm = fast_gradient_method(net, inputs, 0.3, np.inf)
        # x_pgd = projected_gradient_descent(net, inputs, 0.3, 0.01, 40, np.inf)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = Variable(inputs.cuda(), requires_grad=True)
        labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #apply the cg steihaug here!
        #cg_steihaug(loss, inputs, outputs)
        grad_f, = cg_steihaug(loss, inputs, net)
        
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
            # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        writer.add_scalar('Training loss',running_loss, (epoch)*12500+i)
        running_loss=0
print('Finished Training')


