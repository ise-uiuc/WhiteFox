
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(117)
        self.tanh = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(117, 38, 1, stride=1, padding=0)
    def forward(self, x0580):
        v1 = self.bn(x0580)
        v2 = torch.relu(v1)
        v3 = self.tanh(v2)
        v4 = self.conv(v3)
        v5 = v4 * 0.5
        v6 = v4 * v4
        v7 = v6 * v4
        v8 = v7 * 0.044715
        v9 = v4 + v8
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v5 * v12
        return v13
# Inputs to the model
x0580 = torch.randn(1, 117, 58, 80)
