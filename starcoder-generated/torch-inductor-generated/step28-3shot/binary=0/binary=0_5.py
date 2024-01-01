
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 1, stride=1, padding=3)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = 2
        v3 = v1 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
