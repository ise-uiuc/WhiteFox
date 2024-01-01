
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer = torch.nn.Conv2d(1, 1, 2)
        self.batch = torch.nn.BatchNorm2d(1, affine=True)
    def forward(self, input, x):
        y = self.convlayer(x)
        z = self.convlayer(1) + self.convlayer(input)
        y = self.batch(y)
        z = self.batch(z)
        return y, z
# Inputs to the model
input = torch.randn(2, 1, 10, 10)
x = torch.randn(2, 1, 10, 10)
