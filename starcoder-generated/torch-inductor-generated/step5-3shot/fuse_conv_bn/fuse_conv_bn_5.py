
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        m0 = nn.Conv2d(2, 4, 3, stride=(2,4))
        self.model = nn.Sequential(m0, nn.BatchNorm2d(4), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
