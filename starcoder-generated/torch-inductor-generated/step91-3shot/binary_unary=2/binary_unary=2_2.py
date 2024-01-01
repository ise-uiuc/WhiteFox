
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=2, padding=0, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(16, 64, 5, stride=1, padding=2, dilation=2, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 - 64
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
