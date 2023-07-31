#!/bin/python3
import onnx 
import torch
import torchvision 
from torch.autograd import Variable
import torch.nn as nn
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



modelfile = './mnist_cnn.pt'
model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()
# define the name of further converted model
onnx_model_name = "lenet.onnx"
# generate model input to build the graph
generated_input = Variable(
    torch.randn(1, 1, 28, 28)
)
# model export into ONNX format
torch.onnx.export(
    model,
    generated_input,
    onnx_model_name,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)


















