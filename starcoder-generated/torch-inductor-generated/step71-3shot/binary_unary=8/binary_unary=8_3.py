
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv(x1)
        v6 = v5 + v4
        v7 = torch.relu(v6)
        v8 = self.conv(x1)
        v9 = v8 + v7
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1024, 25, 25)
