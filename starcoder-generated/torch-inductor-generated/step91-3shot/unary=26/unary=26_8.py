
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(167, 95, 8, stride=5, padding=2, bias=False)
    def forward(self, x107):
        v1 = self.conv_t(x107)
        v2 = v1 > 0
        v3 = v1 * 0.9135688
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x107 = torch.randn(65535, 167, 13, 43)
