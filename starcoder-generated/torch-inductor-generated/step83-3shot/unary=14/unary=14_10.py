
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(4, 7, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 5, 2)
