
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 5, stride=2, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(18, 24, 5, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 2.0
        v3 = F.relu(v2)
        v4 = torch.cat([x1, v3], 1)
        v5 = self.conv2(v4)
        v6 = F.relu(v5)
        v7 = torch.cat([v3, v6], 1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
