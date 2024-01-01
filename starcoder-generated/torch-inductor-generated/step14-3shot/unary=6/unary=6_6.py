
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=2) # The input size here is 256 rather than 224, which is a common difference where the kernel size is odd.
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0.0, max=6.0)
        v4 = torch.clamp(v3, min=0.0, max=6.0)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)
