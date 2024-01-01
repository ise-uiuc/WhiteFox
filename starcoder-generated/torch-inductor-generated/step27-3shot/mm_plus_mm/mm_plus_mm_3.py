
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.c1 = torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1, stride=1, padding=0)
    def forward(self, x1, x2, x3, x4):
        x1 = self.c1(x1)
        x2 = self.c1(x2)
        x3 = self.c1(x3)
        x4 = self.c1(x4)
        x5 = x1 + x2
        x6 = x3 + x4
        x7 = x5 + x6
        return x7
# Inputs to the model
x1 = torch.randn(24, 2, 10, 5)
x2 = torch.randn(24, 2, 10, 5)
x3 = torch.randn(24, 2, 10, 5)
x4 = torch.randn(24, 2, 10, 5)
