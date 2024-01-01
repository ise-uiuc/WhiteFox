
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(in_channels=3, out_channels=4, kernel_size=(2, 2,2), stride=(1, 2, 2), padding=(0, 0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 2, 2, 2)
