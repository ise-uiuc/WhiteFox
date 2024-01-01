
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.23, max_value=1.53):
        super().__init__()
        self.mul = torch.nn.Mul()
        self.conv_transpose = torch.nn.ConvTranspose3d(3, 6, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.mul(inputs, 1.210558464050293)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1, 1)
