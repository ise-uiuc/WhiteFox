
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(1604, 91, 7, stride=1, padding=3)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(91, 52, 7, stride=1, padding=3)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(52, 21, 7, stride=1, padding=3)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(21, 1, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_2(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_3(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1604, 104, 104)
