
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        y1 = self.conv(x1)
        y2 = y1 + 3
        y3 = torch.clamp_min(y2, 0)
        y4 = torch.clamp_max(y3, 6)
        y5 = torch.div(y4, 6)
        return y5
# Inputs to the model
b2 = torch.randn(1, 3, 64, 64)
