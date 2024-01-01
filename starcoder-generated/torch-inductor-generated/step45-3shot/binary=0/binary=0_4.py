
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 9, 1, stride=1, padding=1)
    def forward(self, x1, other=False):
        v1 = self.conv(x1)
        if other == 2:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
