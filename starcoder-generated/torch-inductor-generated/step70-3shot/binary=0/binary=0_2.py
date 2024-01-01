
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1, padding1=None, padding2=None, other=None):
        v1 = self.conv(x1)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
