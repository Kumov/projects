#!/usr/bin/python3

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


transformed = transforms.Compose(
    [transforms.Resize(28),transforms.ToTensor(),transforms.Normalize([.5],[.5])])

trainset = datasets.MNIST('data2', download=True, train=True,
    transform=transformed)
valset = datasets.MNIST('data2', download=True, train=True,
    transform=transformed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
print(images.size())
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print(model)

criterion = nn.NLLLoss()
#images, labels = next(iter(trainloader))
#images = images.view(images.shape[0], -1)

#logps = model(images) #log probabilities
#loss = criterion(logps, labels) #calculate the loss
#print('df', model[0].weight.grad)
#loss.backward()
#print('df', model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
      images = images.view(images.shape[0], -1)
    #  print(images.shape[-1])
      optimizer.zero_grad()
      print(images)
      print(images.size())
      output = model(images)
      loss = criterion(output, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
    else:
      print("Epoch {} - Training loss: {}".format(e,running_loss/len(trainloader)))

print("\nTraining Time =", (time()-time0)/60)

correct_count, all_count = 0,0
for images, labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1,784)
    with torch.no_grad():
      logps = model(img)

  ps = torch.exp(logps)
  probab = list(ps.numpy()[0])
  pred_label = probab.index(max(probab))
  true_label = labels.numpy()[i]
  if(true_label == pred_label):
    correct_count += 1
  all_count += 1

print("images tested ", all_count)
print("accuracy ", (correct_count/all_count))

#torch.save(model, './handWriteCS_model.pt')
