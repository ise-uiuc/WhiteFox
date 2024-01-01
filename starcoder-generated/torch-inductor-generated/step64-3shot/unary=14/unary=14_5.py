
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=0, output_padding=0)
        self.conv_1 = torch.nn.Conv2d(3, 2, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
