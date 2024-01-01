
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_256_2 = torch.nn.ConvTranspose2d(256, 256, 2, stride=1, padding=0)
        self.sigmoid_4096_1 = torch.nn.Sigmoid(0, False, True)
    def forward(self, x1):
        v1 = self.conv_transpose_256_2(x1)
        v2 = self.sigmoid_4096_1(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 16, 16)
