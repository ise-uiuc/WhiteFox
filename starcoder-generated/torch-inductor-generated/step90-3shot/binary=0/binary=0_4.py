
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 6, 1, stride=1, padding=1)
    def forward(self, x1, x2, other=None, other1=None):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        if other == None:
            other = torch.zeros(v1.shape)
        v3 = v1 + other
        if other1 == None:
            other1 = torch.randn(v2.shape)
        v4 = v2 + other1
        return torch.cat([v3, v4], dim=1)
# Inputs to the model
x1 = torch.randn(1, 23, 25, 19)
x2 = torch.randn(1, 23, 92, 37)
