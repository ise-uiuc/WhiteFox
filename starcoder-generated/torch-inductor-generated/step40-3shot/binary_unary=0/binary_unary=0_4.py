
class Model(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.conv2(v2)
        v4 = v3 + x3
        v5 = self.conv3(v4)
        v6 = v5 + x4
        v7 = self.conv4(v6)
        v8 = v7 + x5
        v9 = self.conv5(v8)
        v10 = v9 + x6
        v11 = self.conv6(v10)
        v12 = v11 + x1
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
