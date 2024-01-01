
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, 0.7*x, x, y, 1.3*y, x+y], dim=0)
        z = y
        for i in range(3):
            y = torch.cat([y, 0.7*y, y, z, 1.3*z, y+z], dim=0)
            z = y
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(3, 4)
