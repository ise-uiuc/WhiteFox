
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2, 1)
        self.conv = torch.nn.Conv2d(9, 27, 3, stride=1, padding=18)
    def forward(self, x1):
        c1 = self.pool(x1)
        c2 = self.conv(c1)
        c3 = c2 + 3
        c4 = torch.clamp_min(c3, 0)
        c5 = torch.clamp_max(c4, 6)
        c6 = c2 * c5
        c7 = c6 / 6
        return c7
# Inputs to the model
x1 = torch.rand(1, 9, 256, 256)
