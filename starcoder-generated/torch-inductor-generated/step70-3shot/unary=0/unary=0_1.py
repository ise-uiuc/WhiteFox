
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 11, 1, stride=3, padding=16)
        self.conv2 = torch.nn.Conv2d(11, 25, 1, stride=3, padding=3)
        self.conv = self.conv1 + self.conv2
    def forward(self, x66):
        v1 = self.conv1(x66)
        v2 = self.conv2(x66)
        v3 = v1 + v2
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
x66 = torch.randn(1, 2, 52, 47)
