
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(476, 338, 2, stride=1, padding=0, bias=False)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0
        v3 = v1 * 0.4987
        v4 = torch.where(v2, v1, v3)
        v5 = v4 + torch.nn.functional.upsample_bilinear(v4, (13, 31))
        return v4
# Inputs to the model
x5 = torch.randn(13, 476, 6, 32)
