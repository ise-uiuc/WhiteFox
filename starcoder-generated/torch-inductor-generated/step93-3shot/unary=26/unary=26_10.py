
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 1, 1, stride=2, padding=3, bias=False)
    def forward(self, x2):
        a = self.conv(x2)
        b = a > 0
        c = a * -0.0197
        d = torch.where(b, a, c)
        return d
# Inputs to the model
x2 = torch.randn(8, 15, 8, 10)
