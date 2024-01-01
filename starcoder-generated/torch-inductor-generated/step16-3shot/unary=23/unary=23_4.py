
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(128, 128, bias=False, groups=6, kernel_size=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
