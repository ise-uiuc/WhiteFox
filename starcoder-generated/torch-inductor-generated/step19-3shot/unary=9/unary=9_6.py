
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1) + 3
        v3 = torch.clamp_min(v1, 0)
        v5 = torch.div(v3.clamp(max=6), 6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
