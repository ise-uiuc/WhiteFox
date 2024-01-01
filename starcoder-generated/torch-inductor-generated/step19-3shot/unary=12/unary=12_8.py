
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(x1)
        v5 = torch.mul(v3, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
