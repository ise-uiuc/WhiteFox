
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=1, dilation=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * [0.5]
        v3 = v1 * [0.5]
        v4 = v3 * v1
        v5 = v4 * [0.044715]
        v6 = v1 + [0.5]
        v7 = v6 * [0.7978845608028654]
        v8 = torch.stack((v5, v5, v6, v6, v6, v6, v6, v6), 0)
        v9 = v8 + v8
        v10 = torch.tanh(v9)
        v11 = v10 + [1.0000000000000002]
        v12 = v4 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 18, 18)
