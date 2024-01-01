
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(219, 141, 3, stride=1, padding=0, bias=True)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0
        v3 = v1 * -1.5003
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x5 = torch.randn(30, 219, 9, 16)
