
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, stride=4)
        self.conv2 = nn.Conv2d(2, 2, 3, stride=4)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(2)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv2(x)
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
model = Model()
