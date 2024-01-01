
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = v2 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = v1 * v4
        v8 = v2 * v6
        t1 = v7 + v8
        return t1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
