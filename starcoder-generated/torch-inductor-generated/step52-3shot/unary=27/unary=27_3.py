
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, 1)
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, min=-1, max=1)
        v3 = self.bn(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 20, 20)
