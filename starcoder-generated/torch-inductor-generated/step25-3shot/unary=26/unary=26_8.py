
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transposes = torch.nn.ConvTranspose1d(5, 6, 3, stride=1)
        self.negative_slope = negative_slope
    def forward(self, x5):
        v1 = self.conv_transposes(x5)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = -0.483
# Inputs to the model
x5 = torch.randn(3, 5, 5)
