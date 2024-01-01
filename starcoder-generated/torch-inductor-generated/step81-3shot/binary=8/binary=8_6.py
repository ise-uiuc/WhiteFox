
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        vcv = self.conv1(x)
        vc = self.conv2(x)
        v1 = vcv + vc
        v2 = v1 + 1.0
        v3 = v2 + self.conv3(x)
        v4 = v1 + v3
        v5 = self.bn1(v3)
        v4 = v4 + v5
        v6 = v4 + 1.0
        v7 = v6 + self.conv4(x)
        v8 = self.bn2(v7)
        s1 = v4.unsqueeze(0) * v8.unsqueeze(0).transpose(0, 2)
        (n, k) = s1.size()[-2:]
        s2 = s1.reshape(n, k, -1).sum(-1).div(k)
        h = self.bn3(s2)
        h = h + 1.0
        h = self.bn4(h)
        return h
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
