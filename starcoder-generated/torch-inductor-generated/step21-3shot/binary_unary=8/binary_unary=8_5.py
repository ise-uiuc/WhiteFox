
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, 4, stride=2),
            torch.nn.Conv2d(1, 1, 4, stride=2, groups=1),
            torch.nn.Conv2d(1, 1, 4, stride=1, groups=1),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = v1 + v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
