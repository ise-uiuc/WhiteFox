
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        v1 = self.conv2(x1)
        t2 = t1 + 3
        v2 = v1 + 3
        t3 = torch.clamp(t2, 0, 6)
        v3 = torch.clamp(v2, 0, 6)
        t4 = t1 * t3
        v4 = v1 * v3
        t5 = t4 / 6
        v5 = v4 / 6
        t6 = t5 + v5
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
