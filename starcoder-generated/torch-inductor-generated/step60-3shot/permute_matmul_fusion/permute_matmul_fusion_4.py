
import math
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, b, x):
        return torch.bmm(b, x)
# Inputs to the model
b = torch.randn(1, 1280 // 16, 768 // 16)
x = torch.randn(1, 768 // 16, 1280)
