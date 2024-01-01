
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 128, 1, bias=False, padding=0), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
        self.block3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(128, 32, 1, bias=False, padding=0), torch.nn.Tanh())
        self.block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(32, 1, 3, padding=1, stride=2), torch.nn.ReLU(inplace=False))
    def forward(self, x1):
        y = self.block0(x1)
        y0 = self.block3(y)
        y1 = self.block1(y0)
        return y1
# Inputs to the model
x1 = torch.randn(1, 256, 8, 8)
