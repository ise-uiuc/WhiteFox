
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, x3=None, x4=None, x5=None, x6=None, padding1=None, padding2=None):
        v1 = self.conv(x1)
        if x2 is None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        if x3 is None:
            x3 = torch.randn(v2.shape)
        v3 = v2 + x3
        if x4 is None:
            x4 = torch.randn(v3.shape)
        v4 = v3 + x4
        if x5 is None:
            x5 = torch.randn(v4.shape)
        v5 = v4 + x5
        if x6 == None:
            x6 = torch.randn(v5.shape)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
