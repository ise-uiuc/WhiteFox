
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0_conv1 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
        self.block0_bn1 = torch.nn.BatchNorm2d(3, 0.8999999761581421, 0.0, True)
        self.block0_conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.block0_bn2 = torch.nn.BatchNorm2d(3, 0.8999999761581421, 0.0, True)
        self.block0_conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.block0_bn3 = torch.nn.BatchNorm2d(3, 0.8999999761581421, 0.0, True)
    def forward(self, x1):
        v1 = self.block0_conv1(x1)
        v2 = self.block0_bn1(v1)
        v2 = F.relu(v2)
        v1 = self.block0_conv2(v2)
        v2 = self.block0_bn2(v1)
        v2 = F.relu(v2)
        v3 = self.block0_conv3(v2)
        v4 = self.block0_bn3(v3)
        v5 = F.max_pool2d(v4, 2, 2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 288, 288)
