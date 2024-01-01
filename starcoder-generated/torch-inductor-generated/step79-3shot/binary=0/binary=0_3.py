
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=2, padding=1)
    def forward(self, x1, scale=0.5, other=1):
        v1 = self.conv(x1)
        if scale == 0.5:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
