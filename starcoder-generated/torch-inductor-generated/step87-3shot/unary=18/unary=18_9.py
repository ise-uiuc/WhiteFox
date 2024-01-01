
class Model(torch.nn.Module):
    def __init__(self):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 5, 9, 1, 0)
        self.conv2 = torch.nn.Conv2d(5, 7, 11, 1, 0)
        self.conv3 = torch.nn.Conv2d(7, 8, 8, 1, 0)
        self.conv4 = torch.nn.Conv2d(8, 3, 6, 1, 0)
        self.conv5 = torch.nn.Conv2d(3, 8, 6, 1, 0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 56, 56)
