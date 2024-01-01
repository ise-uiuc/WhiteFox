
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=4)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + 3 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v1 / 6
        t1 = 3 + v7
        v8 = self.conv1(t1)
        v9 = self.conv2(v8)
        v10 = v8 + 3 + v9
        v11 = torch.clamp_min(v10, 0)
        v12 = torch.clamp_max(v11, 6)
        v13 = v8 * v12
        return v12, v13
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
