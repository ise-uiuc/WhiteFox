
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.bn1 = nn.BatchNormX2d(32)
        self.conv2 = nn.ConvX2d(32, 32, 3, stride=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvXd(64, 64, 3)
    def forward(self, x):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        y = F.relu(self.conv4((self.conv3(x))))
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
