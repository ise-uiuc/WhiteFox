
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise1 = torch.nn.Conv2d(1, 1, 9, groups=1, padding=4, bias=False)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.pointwise2 = torch.nn.Conv2d(1, 1, 1, groups=1, bias=False)
    def forward(self, x1):
        x2 = self.pointwise1(x1)
        x3 = self.conv1(x2)
        x4 = self.pointwise2(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 1, 1000, 1001)
# Model begins
