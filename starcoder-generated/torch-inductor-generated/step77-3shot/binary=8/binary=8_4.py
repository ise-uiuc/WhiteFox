
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 32, 2, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 32, 2, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 16, 2, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x2)
        v4 = self.conv4(x2)
        v5 = self.conv5(x1)
        v6 = v1 + v4
        v7 = v2 + v3
        v8 = v4 + v5
        v9 = v5 + v6
        v10 = v7 + v8
        v11 = v8 + v9
        v12 = v9 + v7
        v13 = v11 + v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
