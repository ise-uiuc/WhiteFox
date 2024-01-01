
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 220, (1, 1), stride=1, padding=0, use_bias=False)
        self.bn1 = torch.nn.BatchNorm2d(220)
        self.conv2 = torch.nn.Conv2d(220, 64, (1, 5), stride=1, padding=(0,1))
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 12, (1, 1), stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(12)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.permute(0, 2, 3, 1)
        v3 = self.bn1(v2)
        v3 = v3.permute(0, 3, 1, 2)
        v4 = self.conv2(v3)
        v5 = v4.permute(0, 2, 3, 1)
        v6 = self.bn2(v5)
        v6 = v6.permute(0, 3, 1, 2)
        v7 = self.conv3(v6)
        v8 = v7.permute(0, 2, 3, 1)
        v9 = self.bn3(v8)
        v9 = v9.permute(0, 3, 1, 2)
        v10 = v1 + v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 16, 20)
