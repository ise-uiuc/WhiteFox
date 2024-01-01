
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = v1 * 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = self.conv(v4)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 128, 128)
