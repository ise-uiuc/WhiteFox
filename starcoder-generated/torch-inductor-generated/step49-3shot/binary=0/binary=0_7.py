
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(48, 4, 1, stride=1, padding=1)
    def forward(self, x1, other, other1=None):
        v1 = self.conv(x1)
        if other1 is not None:
            v2 = other1
        else:
            v2 = torch.randn(1, 8, 64, 64)    
        v2 += v1
        other = other // v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 48, 64, 64)
other1 = 1
other = other1
