
class Model(nn.Module):
    def __init__(self, groups=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 3, 1, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0, groups=groups)
        self.conv3 = nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0, groups=groups)
        self.convt5 = nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
        self.conv6 = nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv7 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(9, 3, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.convt2(v1)
        bn1 = self.bn1(v2)
        v3 = self.conv3(bn1)
        v4 = self.conv4(v2)
        v5 = self.convt5(v3)
        bn2 = self.bn2(v5)
        v6 = v4 + bn2
        v7 = self.relu(v6 + x3)
        v8 = self.conv6(v7)
        v9 = self.conv7(v6)
        v10 = self.conv8(v8)
        v11 = torch.cat((v9, v10), 1)
        v12 = self.conv9(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
