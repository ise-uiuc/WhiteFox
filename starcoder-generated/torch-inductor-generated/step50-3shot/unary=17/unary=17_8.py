
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose3d(1, 1, 4, padding=1, stride=1), torch.nn.ReLU(inplace=False))
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, padding=1, stride=1), torch.nn.ReLU(inplace=False))
    def forward(self, x1):
        y1 = self.block0(x1)
        y2 = self.block1(y1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32, 32)
