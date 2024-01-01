
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 3, (3, 2), 1, (3, 2), 2, 2, True)
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = v1 > 0
        v3 = v1 * -0.25
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 12, 4, 4)
x2 = torch.randn(2, 12, 4, 4)
