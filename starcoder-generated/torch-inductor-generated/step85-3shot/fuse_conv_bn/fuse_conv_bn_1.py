
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_a = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, groups=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2_a = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4_a = torch.nn.Conv2d(64, 64, 3, padding=1, groups=2)
        self.bn4 = torch.nn.BatchNorm2d(64)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1_a(x)))
        y = F.relu(self.bn2(self.conv2_a(x)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4_a(y)))
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
