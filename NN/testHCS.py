#!/usr/bin/python3

import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image

image = Image.open('./images/3/3.jpg')
image.show()

PATH = "./images/3/"
predicted_numbers = [3]
transformed = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize([.5],[.5])])

image_data = torchvision.datasets.ImageFolder(root = PATH, transform=
    transformed)
image_loader = data.DataLoader(image_data, batch_size = 1, shuffle = False )

the_model = torch.load('./handWriteCS_model.pt')
the_model.eval()

for images, labels in image_loader:
  for i in range(len(labels)):
    img = images[i].resize(1,784)
    with torch.no_grad():
      logps = the_model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    prediction = probab.index(max(probab))
    print("Picture ", i, ": predicted",  prediction, "...actually answer ",
        predicted_numbers[i])





