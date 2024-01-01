
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=2, groups=2, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.permute(0,2,1,3).gelu()
        v3 = v2.permute(0,2,3,1)
        return v3 * v1
# Inputs to the model
x1 = torch.randn(1, 32, 10, 14)
