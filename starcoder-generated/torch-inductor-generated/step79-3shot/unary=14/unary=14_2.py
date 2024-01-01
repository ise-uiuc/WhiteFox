
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 128, 2, stride=(2, 2), padding=(1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(128, 432, 2, stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
