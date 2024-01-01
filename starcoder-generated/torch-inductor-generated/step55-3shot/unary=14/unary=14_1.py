
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose3d(3, 196, kernel_size=(1, 27, 27), stride=(1, 1, 1), padding=(0, 13, 13))
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224, 224)
