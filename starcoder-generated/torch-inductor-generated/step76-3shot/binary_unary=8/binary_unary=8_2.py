
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 9, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 6, 1, stride=1, padding=2, dilation=2, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
