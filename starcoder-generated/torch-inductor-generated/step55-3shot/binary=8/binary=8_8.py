
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 10, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(10, 6, 3, stride=2, padding=1, output_padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(4, 7, 1, stride=1, padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(6, 4, 1, stride=1, padding=1)
        self.conv6.bias.data.zero_()
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(12)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = self.conv3(v1 + v4 + x2)
        v6 = self.conv5(x3)
        v7 = self.conv4(v3) + v6
        v8 = torch.cat([v5, v7], dim=1)
        v9 = self.conv6(v8)
        return (v8, v9, v4)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 7, 16, 16)
x3 = torch.randn(1, 4, 16, 16)
