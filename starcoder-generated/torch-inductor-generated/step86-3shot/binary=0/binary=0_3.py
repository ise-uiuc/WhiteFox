
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1000, 8, 1, stride=1, padding=1)
    def forward(self, x=None, other=1, padding1=None):
        v1 = self.conv(x)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1000, 64, 64)
