
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v1 > 0
        v4 = v1 * -0.1
        v5 = torch.where(v3, v1, v4)
        v6 = v2 > 0
        v7 = v2 * -0.1
        v8 = torch.where(v6, v2, v7)
        return v5 + v8
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
