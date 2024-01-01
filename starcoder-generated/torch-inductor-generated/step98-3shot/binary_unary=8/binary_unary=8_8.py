
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(3, 9, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv1(x2)
        v6 = self.conv2(x2)
        v7 = v5 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
