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

def load_data(data_path,batch_size):
	# config import transformation, transfrom into torch tensor and applying normalization (from [0,1] to [-1,1])
	transform = transforms.Compose([ \
		transforms.ToTensor(), \
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) \
		])

	#load CIFAR10 dataset, applying the above transform
	train_set = torchvision.datasets.CIFAR10(root=data_path,train=True,download=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory= True) # workers refers number of threads used

	test_set = torchvision.datasets.CIFAR10(root=data_path,train=False,download=False, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory= True)

	return [train_loader, test_loader]

def train(train_loader,net,epoch,criterion,optimizer,use_cuda,snapshot_path,snapshot_interval=2000):
	running_loss = 0.0
	for i, data in enumerate(train_loader, 0): #enumerate starts from 0th item
		# get the inputs
		inputs, labels = data

		# wrap them in Variable
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.data[0] # for average running loss over 2000 mini-batches
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[epoch: %d, batch: %5d/%5d] loss: %.3f' % \
				(epoch + 1, i + 1, len(train_loader.dataset)/train_loader.batch_size,running_loss / 2000))
			running_loss = 0.0
		
		# save snapshot
		if i % snapshot_interval == snapshot_interval-1:
			print('snapshot save at epoch: %d, batch: %d' %\
				(epoch + 1, i + 1))

			snapshot = {'epoch': epoch + 1, \
			'state_dict': net.state_dict(), \
			'optimizer': optimizer.state_dict()}
			torch.save(snapshot, snapshot_path + '/snapshot' + str(epoch+1) + '_' + str(i+1))

	print('Finished epoch %d' % (epoch+1))

def show_image(data_loader,image_num,classes):
		img = np.transpose(data_loader.dataset[image_num][0].numpy(2,0,1))
		img = np.rot90(img,k=3)
		img = img / 2 + 0.5     # unnormalize
		plt.imshow(img)
		plt.title('image: %d, %s' % (image_num, str(classes[data_loader.dataset[image_num][1]])))
		plt.axis('off')
		plt.show()

def main():
	batch_size = 4
	[train_loader, test_loader] = load_data('./CIFAR10_data',batch_size)
	use_cuda = True
	snapshot_folder = './convnet_CIFAR10_snapshot'
	training = True
	snapshot_interval = 2000 # batch interval
	load_snapshot = True
	snapshot_file = 'snapshot2_2000'

	print('Finish loading data')
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	show_random_image = False
	if show_random_image:
		# randomly show few pictures
		for i in range(3):
			random_pick = random.randint(0,len(train_loader.dataset)-1)
			show_image(train_loader,random_pick,classes)

	# defining neural network
	net = ConvNet()
	if use_cuda:
		net.cuda()

	# defining loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #SGD with momentum

	start_epoch = 1
	if load_snapshot:
		resume_path = snapshot_folder+'/'+snapshot_file
		# net.load_state_dict(torch.load(snapshot_folder+'/'+snapshot_file))
		print("=> Loading snapshot '%s'" % (resume_path))
		snapshot = torch.load(resume_path)
		start_epoch = snapshot['epoch']
		net.load_state_dict(snapshot['state_dict'])
		optimizer.load_state_dict(snapshot['optimizer'])
		print("Net state at epoch %d" % snapshot['epoch'])

	# training
	if training:
		if not os.path.isdir(snapshot_folder):
			os.mkdir(snapshot_folder)

		for epoch in range(2):  # loop over the dataset multiple times, 1 epoch equals to 1 loop over
			if epoch + 1 >= start_epoch:
				if load_snapshot:
					print('Resume training at epoch %d' %snapshot['epoch'])
				print('Training starts at epoch %d' % (epoch + 1))
				train(train_loader,net,epoch,criterion,optimizer,use_cuda,snapshot_folder,snapshot_interval)
		print("Finish training")

if __name__ == '__main__':
	main()