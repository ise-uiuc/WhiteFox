
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.transpose(x1, 1, 2)
        v2 = self.conv(v1)
        v3 = torch.transpose(x1, 1, 2)
        v4 = self.conv(v3)
        v5 = torch.transpose(x1, 1, 2)
        v6 = self.conv(v5)
        v7 = v2 + v4
        v8 = torch.relu(v7)
        v9 = torch.transpose(x1, 1, 2)
        v10 = self.conv(v9)
        v11 = torch.transpose(x1, 1, 2)
        v12 = self.conv(v11)
        v13 = torch.transpose(x1, 1, 2)
        v14 = self.conv(v13)
        v15 = v10 + v12 + v14
        v16 = torch.relu(v15)
        v17 = torch.transpose(x1, 1, 2)
        v18 = self.conv(v17)
        v19 = torch.transpose(x1, 1, 2)
        v20 = self.conv(v19)
        return v18 + v20
# Inputs to the model
x1 = torch.randn(1, 3, 64, 256)
