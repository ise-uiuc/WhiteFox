
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU6(True)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU6(True)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.relu3 = torch.nn.ReLU6(True)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.deconv = torch.nn.ConvTranspose2d(16, 16, 1)
        self.relu4 = torch.nn.ReLU(True)
        self.conv5 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.relu5 = torch.nn.ReLU(True)
        self.conv6 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = self.relu2(v5)
        v7 = self.conv3(v6)
        v8 = self.relu3(v7)
        v9 = self.conv4(v8)
        v10 = self.bn3(v9)
        v11 = self.deconv(v10)
        v12 = self.relu4(v11)
        v13 = self.conv5(v12)
        v14 = self.relu5(v13)
        v15 = self.conv6(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
