
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 8, 2)
        self.norm = torch.nn.BatchNorm2d(8, affine=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
# Inputs to the model
x = torch.randn(1, 7, 6, 4)
