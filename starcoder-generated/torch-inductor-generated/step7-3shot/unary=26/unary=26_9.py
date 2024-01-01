
class Model(torch.nn.Module):
    def __init__(self, stride, padding, negative_slope=0.25, output_padding=0):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, (5, 7), stride, padding, output_padding, 1, 1)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * -0.25
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(8, 19, 8, 8)
