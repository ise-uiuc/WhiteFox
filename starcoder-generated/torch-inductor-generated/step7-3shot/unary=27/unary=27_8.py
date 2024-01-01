
class Model(torch.nn.Module):
    def __init__(self, v=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 4, stride=1, padding=3)
        self.v = v
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.v)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
