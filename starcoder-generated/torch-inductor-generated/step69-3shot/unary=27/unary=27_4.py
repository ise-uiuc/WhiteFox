
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 7, stride=1, padding=0)
        self.min = torch.tensor([min])[0]
        self.max = torch.tensor([max])[0]
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1.
max = 3.8
# Inputs to the model
x1 = torch.randn(1, 32, 738, 329)
