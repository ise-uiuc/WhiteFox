
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=1)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn5 = torch.nn.BatchNorm2d(8)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn6 = torch.nn.BatchNorm2d(8)
        self.conv7 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn7 = torch.nn.BatchNorm2d(8)
        self.conv8 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn8 = torch.nn.BatchNorm2d(8)
        self.conv9 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.bn9 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        bn1 = self.bn1(v1)
        bn2 = self.bn2(v2) 
        v3 = bn1 + bn2
        v4 = self.conv3(v3)
        v5 = self.conv4(v3)
        bn3 = self.bn3(v4)
        bn4 = self.bn4(v5)
        v6 = bn3 + bn4
        v7 = self.conv5(v6)
        v8 = self.conv6(v6)
        bn5 = self.bn5(v7)
        bn6 = self.bn6(v8)
        v9 = bn5 + bn6
        v10 = self.conv7(v9)
        v11 = self.conv8(v9)
        bn7 = self.bn7(v10)
        bn8 = self.bn8(v11)
        v12 = bn7 + bn8
        v13 = self.conv9(v12)
        bn9 = self.bn9(v13)
        return bn9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
