
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v2 = v1 + v1
        if other == None:
            other = v1
        v3 = torch.cat([v2, other], 1)
        v3 = v3.flatten(1, -1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
