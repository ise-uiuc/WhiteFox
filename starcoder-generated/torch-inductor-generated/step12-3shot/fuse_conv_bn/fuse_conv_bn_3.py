
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        return F.conv2d(x, self.conv.weight, None)
# Inputs to the model
x = torch.randn(2, 3, 5, 5)
