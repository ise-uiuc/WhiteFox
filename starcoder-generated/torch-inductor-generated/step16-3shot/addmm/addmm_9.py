
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Conv2d(3, 0, (444, 555), bias=True)
    def forward(self, x1, x2):
        v1 = self.m(x1)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(0, 1, 333, 555)
x2 = torch.randn(1, 0)
