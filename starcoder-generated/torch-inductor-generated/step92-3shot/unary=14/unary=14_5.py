
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(5, 64, 3, stride=3, padding=0, output_padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(15, 8, 4, stride=2, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 16, 16)
