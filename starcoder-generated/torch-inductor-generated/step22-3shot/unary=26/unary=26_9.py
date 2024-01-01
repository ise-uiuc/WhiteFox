
class Model(torch.nn.Module):
    def __init__(self, stride, padding, output_padding):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 16, (2, 2), stride, padding=padding, output_padding=output_padding)
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = v1 > 5.784
        v3 = v1 * 2.428
        v4 = torch.where(v2, v1, v3)
        return v4
stride = 2
padding = 0
output_padding = 0
# Inputs to the model
x3 = torch.randn(16, 15, 8, 8)
