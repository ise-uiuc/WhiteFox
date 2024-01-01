
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, padding1=None):
        if x2 == None:
            x2 = torch.randn(1, 32, 64, 64)
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(x2.shape)
        v2 = x2 + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
