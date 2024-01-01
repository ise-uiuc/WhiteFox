
class Model(torch.nn.Module):
    def __init__(self, output_padding):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=output_padding)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 > 0
        v3 = v1 * 1
        v4 = torch.where(v2, v1, v3)
        return v4
output_padding = 2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
