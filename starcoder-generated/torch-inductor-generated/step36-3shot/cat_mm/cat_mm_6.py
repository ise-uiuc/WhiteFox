
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        # There are two pattern: matmul and conv
        # Here we use matmul as an example
        return F.conv1d(x1, torch.randn(10, 3, 2))
# Inputs to the model
x1 = torch.randn(3, 1, 10)
