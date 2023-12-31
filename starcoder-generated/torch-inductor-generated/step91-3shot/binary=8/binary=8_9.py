
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 12, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 20, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 20, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.conv4 = torch.nn.Conv2d(20, 28, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(20, 28, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(20, 28, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(28)
        self.conv7 = torch.nn.Conv2d(28, 16, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(28, 16, 1, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(28, 16, 1, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(16, 10)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = v1.add(v1.mean([2, 3], True))
        v3 = self.bn1(v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(v3)
        v6 = v4.add_(v5.mean([2, 3], True))
        v7 = self.bn2(v6)
        v8 = self.conv4(v7)
        v9 = self.conv5(v7)
        v10 = self.conv6(v7)
        v11 = v8.add_(v10.mean([2, 3], True))
        v12 = self.bn3(v11)
        v13 = self.conv7(v12)
        v14 = self.conv8(v12)
        v15 = self.conv9(v12)
        v16 = v13.add_(v15.mean([2, 3], True))
        v17 = self.bn4(v16)
        v18 = self.avgpool(v17).flatten(1)
        v19 = self.fc(v18)
        return v19
# Inputs to the model
input = torch.randn(1, 7, 32, 32)
