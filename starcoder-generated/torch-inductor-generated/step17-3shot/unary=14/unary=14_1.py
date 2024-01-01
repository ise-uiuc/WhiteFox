
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose_conv = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.transpose_conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
