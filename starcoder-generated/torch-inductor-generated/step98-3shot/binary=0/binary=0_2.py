
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=2)
    def forward(self, x1, x2, other=None):
        v1 = self.conv(x1)
        if other == None:
            v2 = v1 + x2
        else:
            v3 = v1 + x2
            v2 = v3 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
