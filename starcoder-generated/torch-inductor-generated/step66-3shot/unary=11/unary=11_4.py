
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(222, 111, (3), stride=(2))
        self.conv3 = torch.nn.Conv2d(111, 111, 3, stride=2, groups=3)
        self.conv4 = torch.nn.Conv2d(111, 222, 5, stride=1, groups=92)
        self.conv5 = torch.nn.Conv2d(222, 555, (5), stride=(4), groups=59)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv3(v1)
        v3 = self.conv4(v2)
        v4 = self.conv5(v3)
        v5 = v4 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 222, 64, 64)
