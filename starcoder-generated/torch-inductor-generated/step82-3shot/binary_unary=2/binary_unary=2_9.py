
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.75
        v3 = F.relu(v2)
        v4 = self.conv(v3)
        v5 = v4 - 0.5
        v6 = F.relu(v5)
        v7 = self.conv(v6)
        v8 = v7 - 0.5
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
