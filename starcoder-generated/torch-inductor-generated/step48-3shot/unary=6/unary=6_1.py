
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.conv2(x1)
        y3 = torch.clamp_min(y1 + y2, 0)
        y4 = torch.clamp_max(y3, 6)
        y5 = y4 * y1
        y6 = y5 / 6
        return y6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
