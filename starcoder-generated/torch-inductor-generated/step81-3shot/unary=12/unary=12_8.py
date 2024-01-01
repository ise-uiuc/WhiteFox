
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1, dilation=3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = torch.sigmoid(v1)
        v5 = torch.sigmoid(v2)
        v6 = torch.sigmoid(v3)
        v7 = v1 + v4
        v8 = v2 + v5
        v9 = v3 + v6
        v10 = v1 * v7
        v11 = v2 * v8
        v12 = v3 * v9
        v13 = v10 + v11
        v14 = v12 + v13
        return v14
# Inputs to the model
x1 = torch.randn(1, 4, 100, 100)
