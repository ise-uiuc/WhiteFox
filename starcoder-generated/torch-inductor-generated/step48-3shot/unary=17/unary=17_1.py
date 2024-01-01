
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 16, 3, padding=1, stride=2, bias=False), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(16, 3, 1, bias=False))
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1, stride=2, bias=False), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(16, 3, 1, bias=False))
    def forward(self, x1):
        v1 = self.block0(x1)
        v2 = self.block1(v1)
        v3 = self.block1(v2)
        v4 = self.block1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
