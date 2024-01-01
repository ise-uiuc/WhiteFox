
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1)
    def forward(self, x1, x2, x3):
        v3 = self.conv1(x1)
        v4 = self.conv1(x2) + self.conv2(x3)
        v5 = self.conv1(x2)
        v6 = v3 + v4
        v7 = v5 + v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
