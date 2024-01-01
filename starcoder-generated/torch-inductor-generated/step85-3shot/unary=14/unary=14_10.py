
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(225, 225, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(225)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1):
        x1 = self.conv_transpose1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(32, 225, 8, 8)
