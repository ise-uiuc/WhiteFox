
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(93, 57, 7, stride=2, padding=3, dilation=1)
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(57, 32, 1, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_15(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_17(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 93, 43, 43)
