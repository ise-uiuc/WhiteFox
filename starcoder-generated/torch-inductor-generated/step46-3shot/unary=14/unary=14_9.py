
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(513, 16, 3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 513, 57, 57)
