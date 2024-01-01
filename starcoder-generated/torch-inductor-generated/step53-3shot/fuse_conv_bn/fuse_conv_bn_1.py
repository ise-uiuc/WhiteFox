
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 2, bias=False)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(5, affine=False)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.bn(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
