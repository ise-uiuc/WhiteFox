
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm2d(1, affine=True)
    def forward(self, x):
        x1 = self.norm1(x)
        x2 = self.norm1(x)
        x3 = self.norm1(x)
        return x1 + x2 + x3
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
