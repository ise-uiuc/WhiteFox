
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
    def forward(self, x):
        v1 = F.relu(self.bn1(self.conv1(x)))
        v2 = F.relu(self.bn2(self.conv2(v1)))
        v3 = F.relu(self.conv3(v2))
        v4 = F.relu(self.conv4(v3))
        return v4
# Inputs to the model
x1 = torch.randn(1,3,200,200)
