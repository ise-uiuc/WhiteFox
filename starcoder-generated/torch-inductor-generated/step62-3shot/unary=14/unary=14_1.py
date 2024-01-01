
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(10, 12, 1)
    def forward(self, x1, x2):
        v1 = self.conv_transpose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose1(x2)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = torch.cat((v3, v6), 1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
x2 = torch.randn(1, 10, 64, 64)
