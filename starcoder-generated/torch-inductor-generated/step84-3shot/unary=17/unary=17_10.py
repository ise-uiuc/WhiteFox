
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1)
    def forward(self, x1):
        v2 = torch.clamp(x1, min=-0.5, max=-0.1)
        v1 = self.conv(v2)
        v2 = v1.mul(2.0)
        v3 = v2.floor()
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 10, 10)
