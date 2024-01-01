
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.convV = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, bias=False)
        self.convH = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, bias=False)
        self.conv = lambda x: torch.matmul(self.convV(x), self.convH(x).permute((0, 1, 3, 2)))
        self.bias = torch.nn.Parameter(torch.randn(1))
    def forward(self, x1, padding1=None, padding2='other'):
        if padding1 == None:
            padding1 = self.bias
        if padding2 == 'other':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0), mode='replicate') # ConvH
            padding2 = torch.randn(x1.size())
        x2 = self.conv(x1) + padding2
        x2.abs_() + 1e-6
        x3 = torch.exp(x2) * x1 + 1e-6
        x4 = x3.floor()
        x5 = 10 - (-x3)
        x6 = torch.abs(x3 - x2).max()
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
