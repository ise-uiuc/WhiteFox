

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 7, 8),
            torch.nn.ConvTranspose2d(7, 15, 12),
        )

    def forward(self, input):
        z1 = self.block(input)
        z2 = z1 > 0
        z3 = z1 * -0.385
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
input = torch.randn(10, 3, 22, 35)
