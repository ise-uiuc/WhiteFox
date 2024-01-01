
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 10, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 10, 1, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = 1 + x
        v4 = 1 + v3
        v5 = v2 + v4
        v6 = 1 + x
        v7 = v5 + v6
        v8 = 1 + v7
        v9 = v8 * x
        return v9
# Inputs to the model
x = torch.randn(1, 10, 64, 64)
