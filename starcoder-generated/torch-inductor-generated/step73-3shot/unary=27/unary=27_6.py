
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 30, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=1, padding=1)
        self.b = torch.nn.BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(16, 26, 5, stride=1, padding=1)
        self.b1 = torch.nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = torch.nn.Conv2d(26, 16, 1, stride=1, padding=0)
        self.c = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.c1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.b2 = torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.p = torch.nn.PReLU(num_parameters=16)
        self.p1 = torch.nn.PReLU(num_parameters=16)
        self.r = torch.nn.ReLU(inplace=True)
        self.c2 = torch.nn.Conv2d(16, 30, 1, stride=1, padding=0)
        self.g = torch.nn.Conv2d(16, 30, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.b(v2)
        v4 = self.p(v3)
        v5 = self.conv2(v4)
        v6 = self.b1(v5)
        v7 = self.p1(v6)
        v8 = self.r(v7)
        v9 = self.conv3(v8)
        v10 = self.c(v9)
        v11 = self.c1(v9)
        v12 = self.b2(v11)
        v13 = self.p(v12)
        v14 = self.r(v13)
        v15 = self.c2(v14)
        v16 = self.g(x1)
        v17 = v15 - v16
        return v17
# Inputs to the model
x1 = torch.randn(1, 30, 32, 32)
