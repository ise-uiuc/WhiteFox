
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 1, 4, stride=2, padding=3)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 1, 4, stride=2, padding=3, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
