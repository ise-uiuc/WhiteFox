
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 8, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        s1 = self.relu1(v2)
        v3 = self.conv2(s1)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v4)
        v7 = self.bn2(v6)
        v8 = (self.relu3(v5) + self.relu4(v7)) / 2
        v9 = self.conv5(v8)
        v10 = self.conv6(v9)
        t1 = s1.unsqueeze(0) * v10.unsqueeze(0).transpose(0, 2)
        (n, k) = t1.size()[-2:]
        t2 = t1.reshape(n, k, -1)
        t3 = t2.sum(-1).div(k)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
