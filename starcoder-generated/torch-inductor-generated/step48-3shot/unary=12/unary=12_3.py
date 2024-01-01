
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 9, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(9, 9, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(9, 12, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(9, 16, 3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(9, 17, 3, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v1)
        v4 = self.conv3(v1)
        v5 = self.conv4(v1)
        v6 = torch.sigmoid(v1)
        v7 = torch.mul(v1, v6)
        v8 = torch.sigmoid(v2)
        v9 = torch.mul(v2, v8)
        v10 = torch.sigmoid(v3)
        v11 = torch.mul(v3, v10)
        v12 = torch.sigmoid(v4)
        v13 = torch.mul(v4, v12)
        v14 = torch.sigmoid(v5)
        v15 = torch.mul(v5, v14)
        return v7 + v9 + v11 + v13 + v15
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
