
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.rsqrt(torch.max(v1, dim=[2, 3]))
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 23, 128, 128)
