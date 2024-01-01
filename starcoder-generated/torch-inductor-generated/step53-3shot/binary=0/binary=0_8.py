
x1 = torch.randn(1, 7, 13, 16)
x2 = torch.randn(1, 7, 13, 16)
x3 = torch.randn(1, 13, 8, 8)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(19)
    def forward(self, x1, x2, x3, other=None, padding1=None):
        v1 = self.bn(torch.cat([x1, x2, x3], dim=1))
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 13, 16)
x2 = torch.randn(1, 7, 13, 16)
x3 = torch.randn(1, 13, 8, 8)
