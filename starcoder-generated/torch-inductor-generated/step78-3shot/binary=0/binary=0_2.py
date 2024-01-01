
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 2, stride=2, padding=1)
    def forward(self, x1, other=None, stride1=None):
        if other == None:
            other = torch.randn(x1.shape)
        v1 = self.conv(x1)
        if stride1 == None:
            stride1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
