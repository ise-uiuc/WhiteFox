
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(9, 16, kernel_size=5, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 9, kernel_size=5, stride=2, padding=1, dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v3 = self.conv_transpose2(v1)
        v2 = torch.tanh(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
