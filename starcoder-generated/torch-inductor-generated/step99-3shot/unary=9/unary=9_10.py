
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 4, stride=4, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        t4 = 3
        v2 = v1 + t4
        t5 = torch.clamp_min(v2, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
