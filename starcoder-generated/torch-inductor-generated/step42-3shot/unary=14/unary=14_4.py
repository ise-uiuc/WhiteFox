
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(3, 5, 1, stride=2, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
