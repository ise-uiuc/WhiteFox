
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding2=None, stride2=None):
        v1 = self.conv(x1)
        if other is None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if padding1 == None:
            padding2 = torch.randn(v1.shape)
        v3 = v2 + other
        if stride2 is None:
            stride2 = torch.randn(v1.shape)
        v3 = v3 + stride2
        return v3
# Inputs to the model
x1 = torch.randn(1, 23, 64, 64)
