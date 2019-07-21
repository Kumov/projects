#!/usr/bin/python3

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image

PATH = "./images/test/1.jpg"
#transformed = transforms.Compose(
#    [transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize([.5],[.5])])

#valset = datasets.MNIST('data2', download=True, transform=transformed)

##valset = datasets.ImageFolder(root = PATH,transform=transformed)
#valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

model = torch.load('./handWriteCS_model.pt')
model.eval()


#correct_count, all_count = 0,0
#for images in valloader:
#  for i in range(len(images)):
#    img = images[i].view(1,784)
#    with torch.no_grad():
#      logps = model(img)

#  ps = torch.exp(logps)
#  probab = list(ps.numpy()[0])
#  pred_label = probab.index(max(probab))
#  true_label = labels.numpy()[i]
#  if(true_label == pred_label):
#    correct_count += 1
#  all_count += 1

#print("images tested ", all_count)
#print("accuracy ", (correct_count/all_count))

def process_image(pp):
  img = Image.open(pp)
  img = img.resize((64,784))
  #img = img.transpose((2,0,1))
  #img = img/28
  #img[0] = (img[0] - .5)/.5
  #img[1] = (img[0] - .5)/.5
  #img[2] = (img[0] - .5)/.5

  #img = img[np.newaxis,:]
  image = torch.from_numpy(img)
  image.float()
  print(image)
  return image

def predict(image, model):
  model = model.double()
  with torch.no_grad():
    output = model(image)

  output = torch.exp(output)
  probab = list(output.numpy()[0])
  prediction = probab.index(max(probab))
  print("hot")
  return prediction
  #probs, classes = output.topk(1,dim=1)
  #return probs.item(), classes.item()

image = process_image(PATH)
print(image)
print(model)
bob  = predict(image,model)
print(bob)
