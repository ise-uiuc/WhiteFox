
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=2, bias=False)
    def forward(self, x6):
        v1 = self.conv_t(x6)
        v2 = v1 > 0
        v3 = v1 * -4.94
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x6 = torch.randn(1, 4, 35, 42)
