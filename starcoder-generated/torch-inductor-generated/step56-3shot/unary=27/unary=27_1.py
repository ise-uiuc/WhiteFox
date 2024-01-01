
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.3
max = 0.4
# Inputs to the model
x1 = torch.randn(1, 3, 299, 299)
