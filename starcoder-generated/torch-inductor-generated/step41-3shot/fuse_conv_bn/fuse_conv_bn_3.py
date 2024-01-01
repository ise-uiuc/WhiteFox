
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ConvTranspose2d(3, 3, 2)
        self.b = torch.nn.Batchnorm2d(3, affine=True)
        self.c = torch.nn.Conv2d(3, 3, 2)
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        y = self.c(x)
        return y
# Inputs to the model
x = torch.randn(3, 3, 14, 14)
