
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(58, 29, 6, stride=1, padding=1, output_padding=1, bias=False)
    def forward(self, x7):
        v1 = self.conv_t(x7)
        v2 = v1 > 0
        v3 = v1 * 0.284
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x7 = torch.randn(3, 58, 2, 5)
