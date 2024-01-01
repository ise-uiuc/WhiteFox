
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=True)
        self.batch_norm = torch.nn.BatchNorm2d(3, affine=False)

    def forward(self, x1):
        v1 = F.relu(self.conv(x1))
        v2 = F.batch_norm(x1, None, None, None, None, False, 0, self.conv.weight, self.conv.bias)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
