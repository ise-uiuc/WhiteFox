
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 100, 4, stride=2, padding=1, dilation=1)
        self.conv_ = torch.nn.Conv2d(2, 1, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv_(x1)
        v12 = v2 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
