
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
        self.relu1 = torch.nn.ReLU()
        self.tanh1 = torch.nn.Tanh()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.relu1(v3)
        v5 = self.bn1(v3)
        v6 = self.conv3(x1)
        v7 = self.conv4(x2)
        v8 = v6 + v7
        v9 = self.bn3(v8)
        s1 = v4.unsqueeze(0) * v9.unsqueeze(0).transpose(0, 2)
        (n, k) = s1.size()[-2:]
        s2 = s1.reshape(n, k, -1).sum(-1).div(k)
        z = self.tanh1(s2)
        return (v9, v8, z)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
