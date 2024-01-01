
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(5, 1, (2, 3), stride=(2, 1), padding=0, bias=True)
    def forward(self, x):
        z1 = self.convT(x)
        z2 = z1 > 0.0
        z3 = z1 * 0.165
        z4 = torch.where(z2, z1, z3)
        return z4
# Inputs to the model
x = torch.randn(2, 5, 67, 59)
