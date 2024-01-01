
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2 * x2) * x3
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
x1 = torch.randn(2, 3, 36, 64)
x2 = torch.randn(2, 3, 36, 64)
x3 = torch.randn(4, 3, 36, 64)
