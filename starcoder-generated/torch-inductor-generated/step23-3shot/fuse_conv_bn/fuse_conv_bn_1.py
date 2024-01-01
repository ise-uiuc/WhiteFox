
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x):
        return self.bn(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
