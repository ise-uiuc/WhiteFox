
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.sigm = torch.nn.Sigmoid()
    def forward(self, x1):
        y1 = self.conv(x1)
        y2 = y1 + 3
        y3 = torch.clamp(y2, 0, 6)
        y4 = y1 * y3
        y5 = y4 / 6
        y6 = self.sigm(y5)
        return y6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
