
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x, other=None):
        v1 = self.conv(x)
        if other is None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2

# Inputs to the model
x = torch.randn(1, 3, 80, 80)
