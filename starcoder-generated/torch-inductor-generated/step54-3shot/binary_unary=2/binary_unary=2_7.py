
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 14.5271
        v3 = F.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 - 58.3902
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
