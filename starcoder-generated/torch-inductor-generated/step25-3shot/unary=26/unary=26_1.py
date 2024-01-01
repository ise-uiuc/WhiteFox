
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose1d(24, 50, 4)
        self.negative_slope = negative_slope
    def forward(self, x3):
        v1 = self.convtranspose(x3)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 0.0
# Inputs to the model
x3 = torch.randn(25, 24, 26)
