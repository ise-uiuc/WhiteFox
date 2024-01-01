
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 3, stride=1, padding=1, groups=3)
        self.bn1 = torch.nn.InstanceNorm2d(96)
        self.conv2 = torch.nn.Conv2d(96, 256, 1, stride=1, padding=0)
        self.bn2 = torch.nn.InstanceNorm2d(256)
        self.conv3 = torch.nn.Conv2d(256, 320, 1, stride=1, padding=0)
        self.bn3 = torch.nn.InstanceNorm2d(320)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.bn1(t1)
        t3 = F.relu(t2)
        t4 = self.conv2(t3)
        t5 = self.bn2(t4)
        t6 = F.relu(t5)
        t7 = self.conv3(t6)
        t8 = self.bn3(t7)
        t9 = F.relu(t8)
        t10 = self.pool(t9)
        t11 = torch.flatten(t10, 1)
        return t11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
