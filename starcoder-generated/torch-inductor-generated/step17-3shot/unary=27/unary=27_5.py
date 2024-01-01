
class Model(torch.nn.Module):
    def __init__(self, p=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=0)
        self.p = p
    def forward(self, x1):
        v1 = self.conv(x1)
        if self.p is True:
            v2 = torch.clamp_min(v1, 0.8)
            v3 = torch.clamp_max(v2, 3.5)
            v4 = torch.abs(v3)
        else:
            y = v1 > 0
            z = torch.sum(y)
            v4 = z
        return v4
p = True
# Inputs to the model
x1 = torch.randn(1, 3, 12, 13)
