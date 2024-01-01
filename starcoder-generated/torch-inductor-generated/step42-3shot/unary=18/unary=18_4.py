
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.conv(v10)
        v12 = torch.sigmoid(v11)
        v13 = self.conv(v12)
        v14 = torch.sigmoid(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
