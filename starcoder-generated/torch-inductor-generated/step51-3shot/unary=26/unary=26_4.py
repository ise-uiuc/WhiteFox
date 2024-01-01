
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(18, 7, 5, stride=1, padding=0, bias=False)
    def forward(self, x6):
        u1 = self.conv_t(x6)
        v2 = u1 > 29.0391
        v3 = u1 * 0.4811
        v4 = torch.where(v2, u1, v3)
        return v4
# Inputs to the model
x6 = torch.randn(76, 18, 83)
