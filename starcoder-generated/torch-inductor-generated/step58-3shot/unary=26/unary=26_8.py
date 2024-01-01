
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 46, 9, stride=7, padding=7, bias=False)
    def forward(self, x14):
        v4 = self.conv_t(x14)
        v5 = v4 > 0
        v6 = v4 * 0.36
        v7 = torch.where(v5, v4, v6)
        return v7
# Inputs to the model
x14 = torch.randn(4, 5, 75, 28)
