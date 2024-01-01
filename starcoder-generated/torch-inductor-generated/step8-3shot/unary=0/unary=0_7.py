
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
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
x3 = torch.randn(1, 64, 16, 15)
