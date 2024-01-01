
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 7, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 19, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = v1 * 0.5
        v1 = v1 * v1
        v1 = v1 * v1
        v1 = v1 * v1
        v2 = self.conv2(x1)
        v2 = v2 * 0.5
        v2 = v2 * v1
        v2 = v2 * v1
        v2 = v2 * v1
        v3 = v2 * 0.044715
        v4 = v2 + v3
        v4 = v1 + v4
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v1 + v6
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v9 * v10
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64, 64, 1)
