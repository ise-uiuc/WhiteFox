
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x):
        return self.bn(x)
# Inputs to the model
x = torch.randn(1, 1, 1, 1)
