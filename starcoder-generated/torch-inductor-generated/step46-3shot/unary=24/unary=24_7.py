
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m0 = [64, 64]
        self.conv1 = torch.nn.Conv2d(3, m0[0], (7, 7), stride=2, padding=3, groups=3)
        self.bn1 = torch.nn.Batchsize([m0[0]])
        self.relu1 = torch.nn.ReLU()
        m1 = [m0[0], m0[1]]
        self.conv2 = torch.nn.Conv2d(m0[0], m1[0], (5, 5), stride=1, padding=2, groups=32)
        self.bn2 = torch.nn.Batchsize([m1[0]])
        self.relu2 = torch.nn.ReLU()
        m2 = [m1[0], m1[1]]
        self.conv3 = torch.nn.Conv2d(m1[0], m2[0], (3, 3), stride=1, padding=1, groups=1)
        self.bn3 = torch.nn.Batchsize([m2[0]])
        self.relu3 = torch.nn.ReLU()
        m3 = [m2[0], m2[1]]
        self.conv4 = torch.nn.Conv2d(m2[0], m3[0], (3, 3), stride=1, padding=1, groups=1)
        self.bn4 = torch.nn.Batchsize([m3[0]])
        self.relu4 = torch.nn.ReLU()
    def forward(self, x):
        negative_slope = 0.30584862
        v0 = self.conv1(x)
        v1 = self.bn1(v0)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = self.relu2(v4)
        v6 = self.conv3(v5)
        v7 = self.bn3(v6)
        v8 = self.relu3(v7)
        v9 = self.conv4(v8)
        v10 = self.bn4(v9)
        v11 = self.relu4(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
