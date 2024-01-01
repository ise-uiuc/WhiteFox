
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn1(v1)
        v3 = self.bn2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
