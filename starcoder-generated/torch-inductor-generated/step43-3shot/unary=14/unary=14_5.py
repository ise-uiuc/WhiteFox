
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(665, 56, 3, stride=2, padding=(0, 1), dilation=(1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(665, 315, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 665, 120, 200)
