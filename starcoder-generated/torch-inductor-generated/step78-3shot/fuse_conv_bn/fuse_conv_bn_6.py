
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(16, 16, (3, 3), stride=(2, 1), padding=(1, 2), dilation=(3, 4))
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(16, affine=False, track_running_stats=False)
        self.relu = F.relu
        self.maxpool = F.max_pool2d
        self.flatten = F.flatten
        self.linear = torch.nn.Linear(3200, 10)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.bn(x2)
        x4 = self.relu(x3)
        x5 = self.maxpool(x4)
        x6 = self.flatten(x5)
        x7 = self.linear(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)
