
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convbn1 = torch.nn.BatchNorm2d(2, affine=True)
        self.convbn2 = torch.nn.BatchNorm2d(2, affine=True)
        self.convbn3 = torch.nn.BatchNorm2d(2, affine=True)
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1)
        x3 = x2.permute(2, 1, 0)
        x4 = x3.permute(1, 2, 0)
        x5 = self.convbn1(x1)
        x6 = self.convbn2(x5)
        x7 = self.convbn3(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
