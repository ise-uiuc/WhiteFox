
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2, output_padding=1, bias=False)
        self.negative_slope = -0.0897324098107339
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * self.negative_slope
        x4 = torch.where(x2, x1, x3)
        x5 = x4 * 0.1232023475341797
        x6 = x5 + -0.132448
        return torch.round(x6)
# Inputs to the model
x = torch.randn(1, 128, 8, 8)
