
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(1, 1, 3, groups=1, bias=False)
        self.b = torch.nn.Conv2d(1, 1, 3, stride=1, dilation=1, groups=1, bias=False, padding=1)
    def forward(self, x):
        o1 = self.a(x)
        o3 = self.b(o1)
        return o3
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
