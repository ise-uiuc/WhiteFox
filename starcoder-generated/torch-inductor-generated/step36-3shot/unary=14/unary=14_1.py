
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(77, 163, 7, stride=1, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(163, 29, 5, stride=1, padding=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(29, 1, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 77, 48, 48)
