
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x1, y1):
        x1 = F.max_pool2d(x1, 2, stride=2)
        y1 = F.avg_pool2d(y1, 2, stride=2)
        z0 = torch.cat([x1, y1], 0)
        z1 = self.linear(z0)
        z2 = z1 * 0.7071067811865476
        z3 = torch.erf(z2)
        z4 = z3 + 1
        z5 = torch.abs(z4)
        return z5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
