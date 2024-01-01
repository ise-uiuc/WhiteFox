
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(56, 64, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, other=None):
        if x2 == None:
            x2 = torch.randn(x1.shape)
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 * x2
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(10, 56, 48, 48)
