
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 8, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 8, 1, stride=1, padding=2)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        v1 = 3 + t3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = t3 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
