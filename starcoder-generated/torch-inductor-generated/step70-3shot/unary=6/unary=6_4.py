
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 5, stride=2, padding=2)
    def forward(self, x1):
        p1 = self.conv(x1)
        p2 = p1 + 3
        p3 = torch.clamp(p2, 0, 6)
        p4 = p1 * p3
        p5 = p4 / 6
        return p5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
