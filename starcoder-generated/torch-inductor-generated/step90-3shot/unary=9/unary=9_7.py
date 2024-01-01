
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
    def forward(self, x1):
        y1 = self.conv(x1)
        y2 = y1 + 3
        y3 = y2.clamp(min=0.0, max=6.0)
        y4 = y3.div(6.0)
        return y4
# Inputs to the model
d1 = torch.randn(1, 3, 64, 64)
