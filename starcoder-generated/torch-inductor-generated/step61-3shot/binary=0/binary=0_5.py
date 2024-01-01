
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 13, 2, stride=2, padding=2)
    def forward(self, x1, x2=True, padding1=None, padding2=True):
        v1 = self.conv(x1)
        if x2 == True:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v3 = v2 + padding1
        if padding2 == True:
            padding2 = torch.randn(v2.shape)
        v4 = v3 + padding2
        return v4
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
