
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose3d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 118, 152, 243)
