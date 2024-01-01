
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(7, 512, kernel_size=(1, 1), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose_12(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
