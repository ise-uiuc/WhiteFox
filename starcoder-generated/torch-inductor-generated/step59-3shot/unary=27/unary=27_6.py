
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(25, 15, 3, stride=1, padding=1)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        return v2
min = -0.5
# Inputs to the model
x1 = torch.randn(1, 25, 50, 24)
