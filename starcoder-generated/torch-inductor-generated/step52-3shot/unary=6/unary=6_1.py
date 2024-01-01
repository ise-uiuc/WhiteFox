
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 1, stride=2, padding=0)
    def forward(self, x1):
        y1 = self.conv(x1)
        y2 = y1+3
        y3 = y2.clamp(0, 6)
        y4 = y1 * y3
        y5 = y4 / 6
        return y5
# Inputs to the model
x1 = torch.randn(1, 1, 300, 300)
