
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = [torch.cat(x, 1) for x in [[], [], []]]
        print(len(v)) # len(v) is 3
        return v[0][0]
x = torch.randn(3)
print(Model()(x)) # torch.Size([1])
