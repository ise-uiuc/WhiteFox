
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_837_4 = torch.nn.ConvTranspose2d(256, 64, 16, stride=4, padding=0)
        self.conv_transpose_1981_1 = torch.nn.ConvTranspose1d(477, 64, 4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_837_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1981_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
x2 = torch.randn(1, 477, 10)
