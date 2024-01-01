
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 7), stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, (7, 1), stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool2d((5, 1), stride=3)
        self.conv3 = torch.nn.Conv2d(128, 128, (4, 5), stride=5, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 64, (3, 2), stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 224, (1, 18), stride=2, padding=0)
        self.bn5 = torch.nn.BatchNorm2d(224)
        self.conv6 = torch.nn.Conv2d(224, 13, (1, 1), stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.maxpool(v4)
        v6 = self.conv3(v5)
        v7 = self.bn3(v6)
        v8 = torch.relu(v7)
        v9 = self.conv4(v8)
        v10 = self.bn4(v9)
        v11 = self.conv5(v10)
        v12 = self.bn5(v11)
        v13 = torch.relu(v12)
        v14 = self.conv6(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 1, 705, 200)
