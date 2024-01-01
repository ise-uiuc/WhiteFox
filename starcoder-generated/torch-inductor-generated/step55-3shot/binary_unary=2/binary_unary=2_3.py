
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(22, 26, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(26, 32, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 96, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(96, 128, 1, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 160, 1, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(160, 192, 1, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(192, 224, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1024
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 1300
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 1100
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 1800
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 - 1500
        v15 = F.relu(v14)
        v16 = self.conv6(v15)
        v17 = v16 - 2800
        v18 = F.relu(v17)
        v19 = self.conv7(v18)
        v20 = v19 - 2300
        v21 = F.relu(v20)
        v22 = self.conv8(v21)
        v23 = v22 - 6000
        v24 = F.relu(v23)
        return v24
# Inputs to the model
x1 = torch.randn(2, 22, 62, 62)
