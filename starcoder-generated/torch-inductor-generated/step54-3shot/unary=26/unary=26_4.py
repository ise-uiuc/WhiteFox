
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.ConvTranspose2d(7, 3, 8, stride=2, padding=2, bias=False)
    def forward(self, x):
        x1 = self.c1(x)
        x2 = x1 > 0
        x3 = x1 * -3.2710059
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(1, 7, 87, 109)
