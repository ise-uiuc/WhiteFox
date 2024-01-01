
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_37 = torch.nn.ConvTranspose2d(2271, 256, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_37(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2271, 512, 512)
