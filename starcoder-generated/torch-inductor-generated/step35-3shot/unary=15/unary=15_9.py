
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 8, 2, stride=1, padding=0)
        self.dropout = torch.nn.Dropout2d(0.25, False)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(64, 3)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = torch.nn.functional.interpolate(v1, size=None, scale_factor=2.0, recompute_scale_factor=None, mode='bilinear', align_corners=None)
        v3 = self.conv(v2)
        v4 = self.avgpool(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.dropout(v6)
        v8 = self.conv3(v7)
        v9 = torch.relu(v8)
        v10 = torch.flatten(v9, 1)
        v11 = self.fc(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 144, 256)
