
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.register_buffer('buffer', torch.zeros(3))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.buffer.add(v1)
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3.div(6)
        self.buffer = v4
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
