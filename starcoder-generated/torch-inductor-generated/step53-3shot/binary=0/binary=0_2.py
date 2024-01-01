
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 15, 1, stride=8, padding=0)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(padding1.shape)
        if padding1 == None:
            padding1 = torch.randn(other.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 15, 15)
other = torch.randn(1, 4, 15, 15)
