
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = 3 + self.conv(x1)
        v1 = torch.clamp_min(t1, 0)
        v2 = torch.clamp_max(v1, 6)
        v3 = torch.div(v2, 6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
