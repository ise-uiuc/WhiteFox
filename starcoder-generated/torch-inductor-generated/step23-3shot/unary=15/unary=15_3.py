
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
    def forward(self, x1):
        x = self.conv1(x1)
        x = F.relu(x)
        x = x.mean(1)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
