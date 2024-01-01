
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv(x1))
        v2 = torch.relu(self.conv(v1))
        v3 = torch.relu(self.conv(v2))
        v4 = torch.relu(self.conv(v3))
        v5 = torch.relu(self.conv(v4))
        v6 = torch.relu(self.conv(v5))
        v7 = torch.relu(self.conv(v6))
        v9 = v7 + self.conv(x1)
        return torch.relu(v9)
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
