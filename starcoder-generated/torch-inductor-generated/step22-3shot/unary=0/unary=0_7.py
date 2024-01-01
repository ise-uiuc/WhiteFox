
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 9, 5, stride=3, padding=0)
        self.bn = torch.nn.BatchNorm2d(9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1)
        v3 = torch.flatten(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 11, 6329, 1)
