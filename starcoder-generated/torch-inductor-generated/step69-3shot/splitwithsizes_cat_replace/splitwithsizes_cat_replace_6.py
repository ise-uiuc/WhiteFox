
import torch
from op_test import OpTest, randomize_probability
class TestSplitWithSizesCat1(OpTest):
    def setUp(self):
        self.op_type = "split_with_sizes_cat"
        self.set_npu()
        np.random.seed(10)
        input = np.random.random((1, 16, 15, 15)).astype(self.dtype)
        input_sum = input.sum()
        self.inputs = {'input_tensors': input}

        self.split_sizes_array = (2, 1, 36, 36, 4)
        self.split_sizes = list(self.split_sizes_array)
        split_tensors = torch.split(torch.Tensor(input), self.split_sizes_array, dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)

        output = np.array(concatenated_tensor.data.cpu().numpy(), dtype=self.dtype)
        output_sum = output.sum()
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = "npu"
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_split_with_sizes_cat(self):
        self.check_output_with_place(self.place)


import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 96, 3, 1, 1)])
        self.conv1 = torch.nn.Conv2d(96, 3, 3, 1, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, (6, 9, 4), dim=1)
        concatenated_tensor = torch.cat(split_tensors[:], 15)
        return v1 + concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 96, 3, 1, 1)])
        self.conv1 = torch.nn.Conv2d(96, 3, 3, 1, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, (6, 9, 4), dim=1)
        concatenated_tensor = torch.cat(split_tensors[:], 15)
        return v1 * concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)


import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x1 = F.relu(x1, inplace=True)
        x2 = F.relu(x2, inplace=True)
        return torch.cat([x1, x2], dim=1)

# model = Model()

# model.eval()
# model(x)   


import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x1 = self.bn1(x1)
        x1 = self.sigmoid(x1)
        x2 = self.bn2(x)
        x2 = self.sigmoid(x2)
        return torch.cat([x1, x2], dim=1)

# model = Model()

# model.eval()
# model(x)  


import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1,bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        return torch.cat([x1, x2, x2], dim=1)

# model = Model()

# model.eval()
# model(x)  


import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.branch1 = Model1()
        self.branch2 = Model1()
        self.other_features = nn.Sequential()
    def forward(self, x):
        x = self.branch1(x)
        x = self.branch2(x)
        return x
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.branch1 = Model1()
        self.branch2 = Model1()
        self.other_features = torch.nn.Sequential(torch.nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.branch1(x)
        x = self.branch2(x)
        return x
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.branch1 = Model1()
        self.branch2 = Model1()
        self.other_features = torch.nn.Sequential(torch.nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.branch1(x)
        x = self.other_features[0](x)
        x1 = self.branch2(x)
        return x1
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.branch = torch.nn.ModuleList([Model1(), Model1()])
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.other_features = nn.Sequential(torch.nn.ReLU())
    def forward(self, x):
        x = self.branch[0](x)
        x = self.branch[1](x)
        x = self.conv1(x)
        x = self.other_features(x)
        return x
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, 3, 1, 1, bias=False)
        self.branch = torch.nn.ModuleList([Model1(), Model1()])
        self.other_features = nn.Sequential(torch.nn.ReLU())
    def forward(self, x):
        x = self.branch[0](x)
        x = self.other_features(x)
        x = self.branch[1](x)
        x = self.branch[1](x)
        return x
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, 3, 1, 1, bias=False)
        self.branch = torch.nn.ModuleList([Model1(), Model1()])
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.branch[0](x)
        x2 = self.branch[1](x)
        x2 = self.bn(x2)
        x2 = F.relu(x2)
        return torch.cat([x1, x2, x2], dim=1)
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64


import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.branch = torch.nn.ModuleList([Model1(), Model1()])
        self.other_features = nn.Sequential(torch.nn.Sigmoid())
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.other_features((x + x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) - x.permute(0, 2, 3, 1))
        y2 = self.branch[1](x)
        y3 = x + x.permute(0, 2, 3, 1)
        y3 = self.branch[0](y3).permute(0, 3, 1, 2)
        y4 = self.other_features(y3 - x.permute(0, 2, 3, 1)) - torch.mean(y4, dim=1, keepdim=True)
        return torch.cat([y1, y2, y3, y4], dim=1)
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64


import torch
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.branch = torch.nn.ModuleList([Model1(), Model1()])
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.bn(x)
        y1 = F.sigmoid(y1)
        y2 = self.bn(x)
        y2 = F.sigmoid(y2)
        y3 = y1 * x.permute(0, 2, 3, 1)
        y3 = y3.permute(0, 3, 1, 2)
        y4 = y2 * x.permute(0, 2, 3, 1)
        y4 = y4.permute(0, 3, 1, 2)
        return torch.cat([y1, y2, y3, y4], dim=1)
# model = Model()

# model.eval()
# model(x)    # output shape: 1, 96, 64, 64

import tor