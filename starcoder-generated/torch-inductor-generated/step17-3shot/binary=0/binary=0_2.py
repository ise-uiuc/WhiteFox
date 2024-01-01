
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 5, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, padding1=None, padding2=None):
        v1 = self.conv(x1)
        if x2 is None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v3 = v2 + padding1
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        v4 = v3 + padding2
        return v4
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64)
