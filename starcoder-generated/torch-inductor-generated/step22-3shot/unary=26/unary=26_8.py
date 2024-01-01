
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 4, 5, stride=1, padding=2)
        self.negative_slope = 0.01
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > -1.312
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(3, 19, 20, 20)
