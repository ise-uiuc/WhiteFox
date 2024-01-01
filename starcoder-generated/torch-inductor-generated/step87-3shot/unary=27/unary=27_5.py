
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        return v2

min = 0
max = -0.7
# Inputs to the model
x1 = torch.randn(1, 3, 124, 112)
