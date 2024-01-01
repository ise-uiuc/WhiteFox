
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x2):
        v0 = x2.reshape([1, 3, 64, 64])
        v2 = self.conv(v0)
        v5 = torch.div(v2.clamp(min=0.0, max=6.0) - 3.0, 6.0)
        v7 = v5.reshape([1, 8, 33, 33])
        return v7
# Inputs to the model
x2 = torch.randn(1, 3 * 64 * 64)
