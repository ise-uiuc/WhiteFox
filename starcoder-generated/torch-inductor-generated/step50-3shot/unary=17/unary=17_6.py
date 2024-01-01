
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 3, 4, padding=1, stride=1), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
        self.block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 3, 4, padding=1, stride=1), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
    def forward(self, x1):
        y = self.block0(x1)
        y = self.block1(y)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
