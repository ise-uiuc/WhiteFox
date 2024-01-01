
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 79, 3, stride=1, padding=1)
    def forward(self, x1, other=0.5, padding1=None):
        v1 = self.conv(x1)
        if padding1 is None:
            padding1 = torch.randn(v1.shape)
            padding1 = padding1 + other
        v2 = v1 + padding1
        return v2

# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)
