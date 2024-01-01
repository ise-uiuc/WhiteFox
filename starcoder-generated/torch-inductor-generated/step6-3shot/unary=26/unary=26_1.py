
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpos = torch.nn.ConvTranspose2d(3, 12, 5, stride=2, padding=2)
    def forward(self, x2):
        v1 = self.conv_transpos(x2)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
