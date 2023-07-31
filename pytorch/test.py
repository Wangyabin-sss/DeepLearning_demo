#!/bin/python3


import torch
import numpy
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image, ImageStat
import cv2 as cv
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def imag():
    # 调整图片大小
    im = plt.imread('../images/5.jpg')  # 仅作显示用
    images = Image.open('../images/5.jpg')    # 将图片存储到images里面
    images = images.resize((28,28))   # 调整图片的大小为28*28
    images = images.convert('L')   # 灰度化

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    images = transform(images)
    images = images.resize(1,1,28,28)#处理完毕
    print(images)

    # 加载网络和参数
    model = Net()#加载模型
    model.load_state_dict(torch.load('mnist_cnn.pt'))#加载参数
    #model.load_state_dict(torch.load('/home/ubuntu/Desktop/deeplearning/pytorch/001手写数字识别/mnist_cnn.pt'))#加载参数
    model.eval()#测试模型
    outputs = model(images)#输出结果
    print(outputs)
    print(torch.exp(outputs))

    label = outputs.argmax(dim =1) # 返回最大概率值的下标
    plt.title('{}'.format(int(label)))
    plt.imshow(im)
    plt.show()

imag()





