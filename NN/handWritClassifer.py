from torchvision import transforms

#transforms are image transformations
#compose puts multiple transforms together
#first param in Compose takes the raw data and converts it into tensor
#variables (basicallt y turns images to numbers
#second param normalizes it
#normalize, .5 for each color channels of RGB
#do it so the data can be within a specific range so it can train faster
#keeps weights near zero

_tasks = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((.5,.5,.5), (.5,.5,.5))])

from torchvision.datasets import MNIST

#Loads MNIST dataset and applies transformations
mnist = MNIST("data", download=True, train=True, transform=_tasks)

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#creates traning and validation split
split = int(.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

#creates sampler objects
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

#creats iterator objects to train and validate datasets
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = nn.Linear(784, 128)
    self.output = nn.Linear(128, 10)

  def forward(self, x):
    x = self.hidden(x)
    x = F.sigmoid(x)
    x = self.output(x)
    return x

model = Model()

from torch import optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.01, weight_decay= 1e-6, momentum =
    .9, nesterov = True)

for epoch in range(1,11):
  train_loss, valid_loss = [], []

  #train
  model.train()
  for data, target in trainloader:
    optimizer.zero_grad()

    #foward
    output = model(data)
    #find loss
    loss = loss_function(output, target)
    #back
    loss.backward
    #change weights
    optimizer.step()

    train_loss.append(loss.item())


#print eval
model.eval()
for data, target in validloader:
  output = model(data)
  loss = loss_function(output, target)
  valid_loss.append(loss.item())
  print("Trial:" + epoch + "Training Loss: " + np.mean(train_loss) +  "Valid Loss: " + np.mean(valid_loss))


