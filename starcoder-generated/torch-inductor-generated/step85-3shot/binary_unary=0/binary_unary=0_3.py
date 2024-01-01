
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, stride=1, padding=2, groups=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=2)
        self.conv4 = torch.nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv7 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + v2
        v6 = torch.nn.functional.relu(v5)
        u1 = self.conv3(x3)
        u2 = u1 + x1
        u3 = torch.nn.functional.relu(u2)
        u4 = self.conv4(u3)
        u5 = u4 + x1
        u6 = torch.nn.functional.relu(u5)
        u7 = self.conv5(u6)
        u8 = u7 + u7
        u9 = torch.nn.functional.relu(u8)
        u10 = self.conv6(u9)
        u11 = u10 + u2
        u12 = torch.nn.functional.relu(u11)
        u13 = self.conv7(u12)
        u14 = x2 + u13
        u15 = torch.nn.functional.relu(u14)
        return u15
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
