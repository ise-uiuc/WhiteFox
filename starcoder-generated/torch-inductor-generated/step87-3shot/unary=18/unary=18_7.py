
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv2(v2)
        v6 = torch.sigmoid(v5)
        v7 = self.conv2(v2)
        v8 = torch.sigmoid(v7)
        v9 = self.conv2(v2)
        v10 = torch.sigmoid(v9)
        v11 = self.conv2(v2)
        v12 = torch.sigmoid(v11)
        v13 = self.conv2(v2)
        v14 = torch.sigmoid(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 144, 240)
