
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 32, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 8, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 32, 1, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(x1)
        v6 = self.conv6(v5)
        v7 = v4.add(v1)
        v8 = self.conv1(x2)
        v9 = self.conv2(v7)
        v10 = self.conv3(v9)
        v11 = self.conv4(v10)
        v12 = self.conv5(x2)
        v13 = self.conv6(v12)
        return v13 + v11
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 16, 16)
