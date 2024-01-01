
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        y1 = self.conv(x1) + 3
        y2 = torch.clamp(y1, 0, 6)
        y3 = y2 / 6
        return y2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
