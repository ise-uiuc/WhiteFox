
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 7, stride=3, padding=3, dilation=2, groups=5, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=2, padding=3, dilation=2, groups=5)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x1
        v4 = torch.relu(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 56, 56)
x2 = torch.randn(1, 32, 56, 56)
x3 = torch.randn(1, 32, 56, 56)
