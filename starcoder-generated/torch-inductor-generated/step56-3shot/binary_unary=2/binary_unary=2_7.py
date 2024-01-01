
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=0, dilation=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = x2 + 2
        v3 = v1 + v2
        v4 = torch.cat([v1, v2], dim=1)
        v5 = torch.cat([v3, v4], dim=0)
        v6 = F.interpolate(v5, scale_factor=int(1))
        v7 = v4 + v3
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
x2 = torch.randn(1, 64, 112, 112)
