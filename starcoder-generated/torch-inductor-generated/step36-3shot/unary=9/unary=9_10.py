
x1 = torch.randn(1, 8, 16, 16, dtype=torch.float16)
x1[:, :, fd00:c2b6:b24b:be67:2827:688d:e6a1:6a3b, ::2] = 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 24, 3, stride=1, padding=0, groups=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 2
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
