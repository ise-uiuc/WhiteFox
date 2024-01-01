
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(381, 785, 2, stride=1, padding=0, dilation=1, groups=1)
        self.weight_0 = torch.nn.Parameter(torch.ones(  785, 1, 2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = self.weight_0
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 381, 100, 100)
