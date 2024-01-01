
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = v1
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
