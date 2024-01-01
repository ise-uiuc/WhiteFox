
class Model(torch.nn.Module):
    def __init__(self, minimum=0.1, maximum=0.8):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 2, stride=2, padding=1, dilation=2)
        self.minimum = minimum
        self.maximum = maximum
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.minimum)
        v3 = torch.clamp_max(v2, self.maximum)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 100, 100)
