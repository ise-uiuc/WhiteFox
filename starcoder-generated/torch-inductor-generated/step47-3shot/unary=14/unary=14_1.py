
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 7, 7, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv_transpose_1(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x = torch.randn(8, 1, 56, 56)
