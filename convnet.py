from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
	# neural net in pytorch inherits pytorch nn module class
	def __init__(self):
		super(ConvNet, self).__init__()
		# convnet
		# class initation defines convolution, pooling and cross product (fully connected) layers
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120) #linear fully connected layer
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# image forwarding through the net, F is the functional module of pytorch
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x