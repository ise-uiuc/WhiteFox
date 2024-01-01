
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3, stride=0, padding=0, dilation=1)
        self.maxpool = torch.nn.MaxPool3d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.maxpool(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 7, 7, 7)
