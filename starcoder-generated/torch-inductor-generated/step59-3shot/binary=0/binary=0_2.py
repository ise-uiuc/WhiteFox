
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=2, x3=1):
        v1 = self.conv(x1)
        if other == 2:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if x3 == 1:
            v3 = torch.randn(v1.shape)
        else:
            v3 = torch.randn(v2.shape)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
