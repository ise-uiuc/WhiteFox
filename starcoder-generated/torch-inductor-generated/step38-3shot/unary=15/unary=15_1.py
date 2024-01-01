
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 7, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, 8, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, 9, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(256, 256, 10, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(256, 256, 11, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        v11 = self.conv6(v10)
        v12 = torch.relu(v11)
        v13 = self.conv7(v12)
        v14 = torch.relu(v13)
        v15 = self.conv8(v14)
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(2, 3, 320, 320)
