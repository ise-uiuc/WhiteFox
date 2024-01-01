
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 8, 4, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 6, 4, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv_transpose_1(x)
        v2 = self.conv_transpose_2(x)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4, v3
# Inputs to the model
x = torch.randn(2, 3, 2, 2)
