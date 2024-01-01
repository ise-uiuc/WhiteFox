
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, stride=2, padding=1, groups=2)
    def forward(self, x1):
        v2 = self.conv(x1) + 3
        v3 = v2.clamp(min=0, max=6)
        v4 =  v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
