
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.rsqrt(torch.mean(torch.pow(x1, 2)))
        v2 = x1 * v1
        v3 = self.conv(v2)
        v4 = torch.mean(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
