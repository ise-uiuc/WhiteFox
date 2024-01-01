
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1d = torch.nn.ConvTranspose1d(16, 1, 5, stride=1, padding=2, groups=4)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 3, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4):
        self.conv_transpose1d.weight = torch.nn.Parameter(x1)
        self.conv_transpose1d.bias = torch.nn.Parameter(x2)
        v1 = self.conv_transpose1d(x3)
        self.conv_transpose2d.weight = torch.nn.Parameter(x4)
        v2 = self.conv_transpose2d(v1)
        v3 = v2 + x3
        v4 = torch.clamp_min(v3, 3)
        v5 = torch.clamp_max(v4, 4)
        v6 = v5 / 5
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2)
x3 = torch.randn(1, 7, 4)
x4 = torch.randn(5, 7, 2)
