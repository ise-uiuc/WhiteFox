
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x = torch.clamp(self.conv(x1), min=0, max=6)
        y = x / 6
        return y
