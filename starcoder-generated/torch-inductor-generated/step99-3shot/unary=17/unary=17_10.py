
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 2, 3, groups=2, padding=1, stride=3), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
    def forward(self, x1):
        y = self.block0(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
