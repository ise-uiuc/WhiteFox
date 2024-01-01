
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv5 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv6 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv7 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.conv8 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.bn4 = torch.nn.BatchNorm2d(16)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v1_1 = self.conv2(x2)
        v1_2 = v1
        v1_3 = v1_1
        v2 = self.conv3(x1)
        v3 = self.bn1(self.conv4(x2))
        v3_1 = self.bn2(self.conv5(x1))
        v4 = self.conv6(x2)
        v4_1 = self.conv7(x1)
        v5 = self.conv8(x1)
        v5_1 = self.bn3(self.conv8(x2))
        v5_2 = self.bn4(self.conv5(x2))
        out = torch.cat([v1, v2, v3, v4], 1)
        out = torch.cat([out, v4_1, v5, v5_1, v5_2], 1)
        out = self.bn1(out)
        out = self.bn2(out)
        out = self.bn3(out)
        out = self.bn4(out)
        v6 = out + v5
        v7 = out + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
x2 = torch.randn(1, 3, 20, 20)
