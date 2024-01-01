
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(14, 13, 2, stride=7, padding=3, dilation=2, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)
