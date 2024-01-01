
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, (1, 6), stride=1, padding=(0, 3), bias=False)
        self.bn = nn.BatchNorm2d(8, affine=True)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(8, 3, 1, 6)),
            nn.Parameter(torch.randn(8))
        ])
        for idx, p in enumerate(self.weights):
            self.register_parameter('weights'+f'{idx}', p)
        self.bias = b
    def forward(self, x1):
        self.conv.weight = self.weights[0]
        self.bn.weight = self.weights[1]
        self.bn.bias = self.bias
        v2 = x1.flatten(2).permute(2, 0, 1)
        v1 = self.conv(v2)
        v2 = v1 + 3
        v3 = v2.clamp(0, 6)
        v4 = v3.div(6)
        return v4.view(-1, 8, 14, 14)
params = [
    # conv weight
    torch.randn(8, 3, 1, 6),
    # bn weight
    torch.randn(8),
    # bn bias
    torch.tensor(0.0),
]
x1 = torch.randn(4, 3, 64, 64)
