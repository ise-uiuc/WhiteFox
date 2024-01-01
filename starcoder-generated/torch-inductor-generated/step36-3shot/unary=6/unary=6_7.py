
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.b = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        i = torch.clamp_min(x1, 3) + 7
        o = 9 * i + torch.clamp_max(x1, 3)
        return o - self.b(o) + self.a(o)
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
