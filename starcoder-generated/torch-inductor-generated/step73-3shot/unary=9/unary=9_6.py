
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2
        x4 = x3 * 3
        x5 = torch.clamp_min(x4, 0)
        x6 = torch.clamp_max(x5, 6)
        x7 = x6 / 6
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
