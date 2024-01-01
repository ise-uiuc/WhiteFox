
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(3, 1, kernel_size=3, padding=1, stride=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2

x1 = torch.randn(1, 3, 204, 171, 223)
