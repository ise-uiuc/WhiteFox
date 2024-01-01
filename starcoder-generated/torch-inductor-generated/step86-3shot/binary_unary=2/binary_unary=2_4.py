
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 10, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = 100
        v3 = v1 - v2
        v4 = v3 - v1
        v5 = v1 - v4
        v6 = v4 - v5
        v7 = v3 + v2
        v8 = v6 + v7
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
