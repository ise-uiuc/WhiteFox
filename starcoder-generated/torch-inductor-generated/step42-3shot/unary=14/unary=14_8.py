
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 23, 2, stride=2, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16, dtype=torch.float64)
