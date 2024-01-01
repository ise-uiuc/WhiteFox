
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 20, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x4):
        v1 = self.conv_transpose(x4)
        v2 = v1 > 0
        v67 = 0.67
        v3 = v1 * v67
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(6, 10, 8, 8)
