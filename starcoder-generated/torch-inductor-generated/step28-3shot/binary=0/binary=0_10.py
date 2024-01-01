
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, v2, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v4 = v1 + other
        v5 = v4.mean()
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
v2 = 1
