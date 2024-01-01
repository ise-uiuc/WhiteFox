
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        o1 = self.conv(x1)
        o2 = o1 + 3
        o3 = torch.clamp(o2, 0, 6)
        o4 = o3 * o2
        o5 = o4 / 6
        return o5
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
