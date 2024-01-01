
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(16, 480, 3, stride=1, padding=1, output_padding=1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x26):
        t1 = self.conv_t1(x26)
        f1 = t1 * self.negative_slope
        return f1
negative_slope = 0.0
# Inputs to the model
x26 = torch.randn(16, 16, 16, 24)
