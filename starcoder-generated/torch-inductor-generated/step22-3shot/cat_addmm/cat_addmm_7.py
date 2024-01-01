
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = torch.cat([x, x], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
# Model Ends

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_1d = nn.Conv1d(1, 16, kernel_size=1, stride=1)
        self.conv_2d = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        
    def forward(input):
        x = self.conv_1d(input)
        return self.conv_2d(x)


# Model begins
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(x):
        x = torch.cat((x, x), dim=1)
        x = torch.stack((x, x), dim=1)
        return x
# Model Ends

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_1d = nn.Conv1d(1, 16, kernel_size=1)
        self.conv_2d = nn.Conv2d(16, 1, kernel_size=1)
        
    def forward(input):
        x = self.conv_1d(input)
        return self.conv_2d(x)
# Model begins
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(x):
        x = torch.cat((x, x), dim=1)
        return x
# Model Ends

# Model begins
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Linear(2, 4)
    def forward(x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=1)
        return x
# Model Ends

# Model Begins
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.AvgPool2d(2, 2))
        self.layers2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.AvgPool2d(2, 2))
        self.fc = nn.Linear(32*4*4, 84)
    def forward(self, x):
        out1 = self.layers1(x)
        out2 = self.layers2(out1)
        out3 = out2.view(out2.size(0), -1)
        out4 = self.fc(out3)
        return out4
# Model Ends

##