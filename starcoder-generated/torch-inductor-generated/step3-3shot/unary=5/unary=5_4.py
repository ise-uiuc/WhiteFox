
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.3333333333333333
        v3 = v2 * 0.3333333333333333
        v4 = v2 * 0.3333333333333333
        v5 = v1 + v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
