
import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 2, 3, bias=False, groups=1)
        self.batch_norm = torch.nn.BatchNorm2d(2, momentum=0.1)
        self.relu = F.relu
    def forward(self, x):
        o1 = self.relu(self.batch_norm(self.conv2d(x)))
        o2 = F.relu(self.batch_norm(o1))
        return o2
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
