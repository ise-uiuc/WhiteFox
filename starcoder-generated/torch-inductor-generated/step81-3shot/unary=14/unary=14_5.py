
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_26 = torch.nn.ConvTranspose2d(380, 230, 1, stride=1, bias=False)
        self.conv_transpose_28 = torch.nn.ConvTranspose2d(230, 230, 3, stride=2, padding=2, output_padding=1)
        self.conv_transpose_29 = torch.nn.ConvTranspose2d(230, 1, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_26(x1)
        v2 = self.conv_transpose_28(v1)
        v3 = self.conv_transpose_29(v2)
        v4 = torch.sigmoid(v3)
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 380, 83, 89)
