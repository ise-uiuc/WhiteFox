
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = v1 + v2 + v3 + v4 + v5
        v7 = v6 + v6
        v8 = torch.add(v7, v7)
        v9 = v8 + v8
        v10 = torch.add(v9, v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
