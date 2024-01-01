
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 32, 3, stride=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose(x1)
        v2 = v1.flatten(2)
        v3 = v2 / x3.flatten(2).sum((2, 3))
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 100, 150)
x2 = torch.randn(1, 100, 1)
x3 = torch.randn(1, 100)
