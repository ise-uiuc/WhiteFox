
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=256)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=512)
        self.conv3 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=1024)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.bn2(self.conv2(v1))
        v3 = self.conv3(v2)
        v4 = self.bn3(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 224, 224)
