
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.ones(v1.shape)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
