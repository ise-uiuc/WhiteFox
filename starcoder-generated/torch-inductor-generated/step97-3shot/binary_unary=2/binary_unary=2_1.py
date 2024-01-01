
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(10, eps=0.0010000000474974513, momentum=0.8999999761581421)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.949999988079071
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
