
class ConvBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 1, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v1 * 0.5
        v4 = v2 * v3
        v5 = v2 * v2
        v6 = v3 * v2
        v7 = v6 * 0.044715
        v8 = v2 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v2 + v10
        v12 = v4 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)
