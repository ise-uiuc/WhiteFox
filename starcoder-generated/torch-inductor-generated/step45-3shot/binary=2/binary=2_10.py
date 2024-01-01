
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 32, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.sum(dim=[0, 2, 3]) - 0.5
        return v2
# Inputs to the model
x = torch.randn(2, 1, 8, 4, 7)
