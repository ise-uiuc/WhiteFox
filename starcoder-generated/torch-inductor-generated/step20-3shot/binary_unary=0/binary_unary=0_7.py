
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.relu(v3)
        v5 = self.conv(v4)
        v6 = torch.relu(v5)
        v7 = self.conv(v6)
        v8 = torch.relu(v7)
        v9 = self.conv(v8)
        v10 = torch.relu(v9)
        v11 = self.conv(v10)
        v12 = torch.relu(v11)
        v13 = v1 + v12
        v14 = torch.relu(v13)
        v15 = v14 + x2
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 32, 128, 128)
x2 = torch.randn(1, 32, 128, 128)
