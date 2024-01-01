
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = torch.clamp_max(v1, self.min)
        return v3
min = 2
# Inputs to the model
x1 = torch.randn(1, 1, 50, 1000)
