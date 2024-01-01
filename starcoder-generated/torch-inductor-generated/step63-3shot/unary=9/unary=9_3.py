
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.addcmul(self.conv(x1), 0.00390625, self.conv(x1), value=3)
        v2 = v1.clamp(0, 6)
        v3 = v2.div(6)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
