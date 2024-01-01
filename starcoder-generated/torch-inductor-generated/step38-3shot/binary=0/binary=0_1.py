
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=2)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 * -1
        v3 = other * 2
        v4 = v2 + v3
        v5 = v4 + other
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
