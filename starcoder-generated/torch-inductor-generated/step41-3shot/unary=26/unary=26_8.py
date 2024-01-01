
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 19, 5, stride=1, padding=0, bias=False)
    def forward(self, x5):
        u1 = self.conv_t(x5)
        v2 = u1 > 0
        v3 = u1 * 29.2037
        v4 = torch.where(v2, u1, v3)
        return v4
# Inputs to the model
x5 = torch.randn(38, 12, 25, 16)
