
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 4, 1, stride=5, padding=4)
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose2d(x1, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.dilation, self.conv_transpose.groups)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 29, 29)
