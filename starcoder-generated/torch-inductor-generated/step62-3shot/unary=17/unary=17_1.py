
class ResidualModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv1 = torch.nn.Conv2d(32, 32, 3, padding=1, stride=1)
    def forward(self, x1):
        v0 = self.conv0(x1)
        v1 = torch.functional.relu(v0)
        v2 = self.conv1(v1)
        v3 = torch.functional.relu(v2)
        v3 += x1
        return v3
# Inputs to ResidualModule (residual path)
x1 = torch.randn(1, 32, 128, 128)
# Inputs to ResidualModule (regular path)
x2 = torch.randn(1, 32, 128, 128)
