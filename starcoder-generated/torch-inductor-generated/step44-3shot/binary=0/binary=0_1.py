
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 2, stride=2, padding=1)
    def forward(self, x1, other=0):
        v1 = self.conv(x1)
        if other == 0:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(4, 64, 64, 64)
