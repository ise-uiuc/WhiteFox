
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(26, 26, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = v3 + x1
        return v4
# Inputs to the model
x1 = torch.randn(1, 26, 248, 248)
