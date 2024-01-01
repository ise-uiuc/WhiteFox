
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(62, 17, 5, stride=5, padding=3)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(17, 43, 3, stride=5, padding=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(43, 59, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 62, 5, 5)
