
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 100, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sin(v5)
        v7 = self.conv4(v6)
        v8 = torch.tanh(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.conv6(v10)
        v12 = torch.tanh(v11)
        v13 = self.conv7(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
