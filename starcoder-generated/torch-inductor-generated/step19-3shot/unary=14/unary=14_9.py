
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 * v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 70, 70)
