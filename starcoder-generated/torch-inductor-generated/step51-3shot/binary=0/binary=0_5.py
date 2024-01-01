
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(48, 2, 3, stride=1, padding=2)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        if other is None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 48, 32, 32)
other = torch.randn(1, 64, 32, 32)
