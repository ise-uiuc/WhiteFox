
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 32, 2, padding=1, stride=2), torch.nn.ReLU(inplace=True))
        self.block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(32, 64, 3, padding=2, stride=1), torch.nn.ReLU(inplace=False))
        self.block2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(64, 64, 3, padding=1, stride=1), torch.nn.ReLU(inplace=True))
        self.block3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(64, 1, 3, padding=1, stride=1), torch.nn.ReLU(inplace=False), torch.nn.Sigmoid())
    def forward(self, x1):
        x2 = x1
        x3 = self.block0(x2)
        x4 = x3
        x5 = self.block1(x4)
        x6 = x5
        x7 = self.block2(x6)
        x8 = x7
        y = self.block3(x8)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 112, 112)
