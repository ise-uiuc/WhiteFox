
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), bias=False)
    def forward(self, x):
        return self.conv1(x)
# Input to the model
x = torch.randn(1, 3, 224, 224)
