
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(265, 543, 1, stride=1, bias=False)
    def forward(self, x13):
        v1 = self.conv_t(x13)
        v2 = v1 > 0
        v3 = v1 * -0.556844
        v4 = torch.where(v2, v1, v3)
        return v2
# Inputs to the model
x13 = torch.randn(1, 265, 1, 2)
