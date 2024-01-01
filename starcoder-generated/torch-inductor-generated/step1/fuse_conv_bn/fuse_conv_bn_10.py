
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2)
        self.bn = torch.nn.BatchNorm2d(2, affine=False)

    def forward(self, x1):
        x = self.conv(x1)
        v1 = x.detach()
        v2 = self.bn(v1)
        return v1 + v2


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
