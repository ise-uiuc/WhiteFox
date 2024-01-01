
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 3, 2, stride=3, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv_t(x)
        return t1 * self.negative_slope
negative_slope = 0.61
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
