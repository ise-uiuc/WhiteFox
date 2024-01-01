
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x3_2):
        v1 = self.conv(x1)
        a1 = self.conv(x1)
        v2 = v1 + x2
        a2 = a1 + x2
        v3 = torch.relu(v2)
        v4 = a2 + x2
        v5 = torch.relu(v4)
        v6 = self.conv(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        v9 = self.conv(v8)
        v10 = v9 + x3
        v11 = torch.relu(v10)
        v12 = v11 + x3_2
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x3_2 = torch.randn(1, 16, 64, 64)
