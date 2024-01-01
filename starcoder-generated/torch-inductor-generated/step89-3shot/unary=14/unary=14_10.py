
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(4, 3, kernel_size=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = x1.transpose(1,2)
        v3 = torch.sigmoid(v1.abs())
        v4 = v3 * v2
        return v4
# Inputs to the model
x1 = torch.randn(1,4,4,1)
