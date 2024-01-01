
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv5 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv6 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv7 = torch.nn.Conv2d(3, 4, 1, stride=1)
    def forward(self, x1, x2):
        v12 = self.conv6(x1) + self.conv5(x2) + self.conv7(x2)
        v3 = self.conv1(x1)
        v7 = self.conv3(x2) + self.conv2(x1)
        v1 = self.conv4(x2)
        v17 = v7 + v1
        v16 = v12 + v12
        v13 = v3 + v3
        v4 = self.conv5(x1) + self.conv5(x2)
        v10 = v4 + v3
        v14 = v7 + v3
        v5 = self.conv6(x2) + self.conv6(x1)
        v9 = v10 + v13
        v8 = v14 + v5
        v11 = v8 + v9
        v15 = v3 + v11
        return (v16, v15)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
