
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 12, 5, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(24, 36, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(48, 72, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(24, 1, 3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(24, 3, 5, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(48, 24, 2, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(48, 1, 7, stride=1, padding=3)
        self.conv10 = torch.nn.Conv2d(1, 6, 7, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv3(v1)
        v3 = self.conv4(v2)
        v4 = torch.cat((v2, v3), dim=1)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = v4 - 0.5
        v9 = F.relu(v8)
        v10 = self.conv8(v9)
        v11 = v10 - 1.2
        v12 = F.relu(v11)
        v13 = v5 - 0.5
        v14 = F.relu(v13)
        v15 = self.conv9(v14)
        v16 = v15 - 1
        v17 = F.relu(v16)
        v18 = v17 - 0.007
        v19 = F.relu(v18)
        v20 = self.conv10(v19)
        v21 = v20 - 1
        v22 = F.relu(v21)
        v23 = torch.squeeze(v22, 0)
        return v23
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
