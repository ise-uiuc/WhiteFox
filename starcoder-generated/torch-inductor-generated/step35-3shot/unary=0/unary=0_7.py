
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 * 0.5
        v4 = v1 * v1
        v5 = v4 * v1
        v6 = v5 * 0.044715
        v7 = v1 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        v12 = v2 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
