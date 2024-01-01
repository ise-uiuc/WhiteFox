
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=0)
        self.pool = torch.nn.AvgPool2d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = v2 + 6
        v4 = torch.clamp_min(v3, 3)
        v5 = torch.clamp_max(v4, 8)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
