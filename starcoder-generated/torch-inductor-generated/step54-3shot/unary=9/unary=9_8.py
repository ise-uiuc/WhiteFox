
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp_max(6) - 3
        v4 = v3.div(24)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
# Input to the model ends
