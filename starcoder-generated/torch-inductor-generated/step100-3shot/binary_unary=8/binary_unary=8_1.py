
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 32, 9, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v7 = self.conv(x1)
        v8 = self.conv(x1)
        v9 = self.conv(x1)
        v10 = self.conv(x1)
        v11 = self.conv(x1)
        v12 = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], 1)
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 8, 48, 32)
