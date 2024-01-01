
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, padding=3, dilation=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 125.234
        v4 = F.relu(v3)
        return v4 + self.conv1(x1)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
