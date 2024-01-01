
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, (1, 5), stride=1, padding=(0, 1), dilation=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = 3 + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        v1 = self.conv2(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = v6 + t6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
