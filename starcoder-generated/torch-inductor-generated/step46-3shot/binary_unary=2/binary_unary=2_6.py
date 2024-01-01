
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, 2, stride=2, padding=1, dilation=1)
        self.conv1 = torch.nn.Conv2d(4, 12, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 12, 2, stride=2, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv1(x2)
        v3 = self.conv2(x3)
        v4 = v1 + v2 + v3
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 32, 32)
x3 = torch.randn(1, 5, 16, 16)
