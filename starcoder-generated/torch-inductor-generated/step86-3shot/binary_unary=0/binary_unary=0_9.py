
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv4 = torch.nn.ConvTranspose2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv5 = torch.nn.ConvTranspose2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv6 = torch.nn.ConvTranspose2d(16, 16, 7, stride=1, padding=3, groups=16)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - v6
        v9 = self.conv4(v6)
        v10 = v8 - v9
        v11 = torch.relu(v10)
        v12 = self.conv5(x3)
        v13 = v12 - v10
        v14 = self.conv6(x2)
        v15 = v14 - v13
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
