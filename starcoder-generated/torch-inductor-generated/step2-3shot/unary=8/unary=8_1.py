
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 4, stride=4)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = 3 + v1
        v3 = 0 + v2
        v4 = 6 + v3
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
