
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_32 = torch.nn.ConvTranspose2d(384, 256, 5, stride=3, padding=0)
        self.conv_transpose_122 = torch.nn.ConvTranspose2d(256, 384, 5, stride=(3, 2), padding=(2, 2), dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_32(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_122(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 384, 3, 3)
