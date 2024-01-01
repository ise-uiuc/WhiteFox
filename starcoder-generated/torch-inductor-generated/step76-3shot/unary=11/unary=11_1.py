
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(30, 96, 3, stride=1, padding=1, dilation=5, groups=3)
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = self.conv2d(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.relu(v3)
        v5 = v4 * v2
        v6 = v5 + 3
        v7 = torch.clamp_min(v6, 0)
        v8 = torch.clamp_max(v7, 11)
        v9 = v8 / 22
        return v9
# Inputs to the model
x1 = torch.randn(1, 30, 64, 64)
