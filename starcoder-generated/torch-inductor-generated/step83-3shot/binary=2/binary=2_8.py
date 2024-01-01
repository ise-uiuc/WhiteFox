
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 1), (1, 1), (1, 1))
        self.conv2 = torch.nn.Conv2d(8, 6, (3, 3), (2, 3), (0, 1))
    def forward(self, x ):
        v = self.conv1(x)
        t = self.conv2(x)
        z1 = v - t
        z2 = v / t
        z3 = torch.sum(v / t)
        z4 = v // t
        z5 = v % t
        z6 = v ** 2
        return torch.cat((z1, z2, z3, z4, z5, z6), 0)
# Inputs to the model
x = torch.randn(16, 3, 2, 2)
