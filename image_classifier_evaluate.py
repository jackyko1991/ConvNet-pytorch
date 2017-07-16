import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import os
from convnet import ConvNet

def load_data(data_path):
	# config import transformation, transfrom into torch tensor and applying normalization (from [0,1] to [-1,1])
	transform = transforms.Compose([ \
		transforms.ToTensor(), \
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) \
		])

	dataset = torchvision.datasets.CIFAR10(root=data_path,train=False,download=False, transform=transform)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=3, pin_memory= True)

	return data_loader

def main():
	net = ConvNet()
	net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #SGD with momentum

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	# load snapshot
	snapshot = torch.load('./convnet_CIFAR10_snapshot/snapshot2_12000')
	net.load_state_dict(snapshot['state_dict'])
	optimizer.load_state_dict(snapshot['optimizer'])
	
	data_loader = load_data('./CIFAR10_data') 

	image_num = random.randint(0,len(data_loader.dataset)-1)
	img = np.transpose(data_loader.dataset[image_num][0].numpy(2,0,1))
	img = np.rot90(img,k=3)
	img = img / 2 + 0.5     # unnormalize
	print(classes[data_loader.dataset[image_num][1]])
	plt.imshow(img)
	plt.title('image: %d, %s' % (image_num, str(classes[data_loader.dataset[image_num][1]])))
	plt.axis('off')
	# plt.show()

	img_size = data_loader.dataset[image_num][0].size()
	input = torch.FloatTensor(1, img_size[0], img_size[1],img_size[2]).zero_()
	input[0,:] = data_loader.dataset[image_num][0]

	output = net(Variable(input.cuda()))
	print(output)
	_, predicted = torch.max(output.data, 1)
	print('Predicted: %s' % classes[predicted[0][0]]) # remind that the input has a 4D dimension (batch, channel, width, height)

if __name__ == '__main__':
	main()