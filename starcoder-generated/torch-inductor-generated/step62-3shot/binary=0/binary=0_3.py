
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
    def forward(self, x1, x2, other=True):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        if other == True:
            other = torch.randn(v1.shape)
        v3 = v1 + v2
        t1 = v3 + other
        t2 = t1 + other
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
