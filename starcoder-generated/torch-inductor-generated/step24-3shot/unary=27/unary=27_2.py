
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 9, 9, stride=3, padding=3)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 96, 96)
