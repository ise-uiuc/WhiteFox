
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose3d(2, 5, 3, padding=1, stride=1), torch.nn.ReLU(inplace=True))
        self.conv0 = torch.nn.ConvTranspose2d(6, 1, 3, padding=1, stride=1)
    def forward(self, x1):
        z = self.block0(x1)
        y = self.conv0(z)
        return y
# Inputs to the model
x1 = torch.randn(2, 2, 8, 8, 8)
