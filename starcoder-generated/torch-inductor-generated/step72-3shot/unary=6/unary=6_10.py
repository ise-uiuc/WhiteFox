
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 2, stride=1, padding=0)
    def forward(self, x1):
        d = self.conv(x1)
        e = torch.clamp_min(d, 0)
        f = torch.clamp_max(e, 6)
        g = torch.nn.MaxPool2d(3, stride=1, padding=1)(f)
        h = g / 6
        return h
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
