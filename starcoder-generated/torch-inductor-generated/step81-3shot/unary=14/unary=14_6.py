
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_94 = torch.nn.ConvTranspose3d(2, 20, 12, stride=2, padding=2, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_94(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16, 16)
