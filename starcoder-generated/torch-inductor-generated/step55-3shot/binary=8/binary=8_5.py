
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 24, 1, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(24)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.bn5 = torch.nn.BatchNorm2d(24)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v2)
        v6 = v4.mul(v5)
        v7 = self.conv3(x3)
        v8 = self.conv4(x1)
        v9 = v7 + v8
        v10 = self.bn3(v9)
        v11 = self.bn4(v7)
        m = v10.mul(v11)
        v12 = self.bn5(v2)
        s1 = v12.mul(m)
        (n, k) = s1.size()[-2:]
        s2 = s1.reshape(n, k, -1).sum(-1).div(k)
        return (v3, v8, v12, s2)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
x3 = torch.randn(1, 3, 224, 224)
