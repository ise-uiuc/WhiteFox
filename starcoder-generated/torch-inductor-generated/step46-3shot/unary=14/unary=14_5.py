
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 7, 2, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        t1 = self.conv_transpose_2(x1)
        return t1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
