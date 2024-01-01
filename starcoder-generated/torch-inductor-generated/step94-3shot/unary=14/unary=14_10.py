
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(202, 48, 2, stride=1, padding=2)
        self.conv2d_1 = torch.nn.Conv2d(48, 97, 5, stride=1, padding=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(97, 16, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2d_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_2(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 202, 32, 32)
