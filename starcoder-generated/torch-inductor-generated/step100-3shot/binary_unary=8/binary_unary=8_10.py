
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv8 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v7 = self.conv(x1)
        v8 = self.conv(x1)
        v9 = self.conv(x1)
        v10 = self.conv(x1)
        v11 = self.conv(x1)
        v12 = self.conv(x1)
        v13 = self.conv(x1)
        v14 = self.conv(x1)
        v15 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
