
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(251, 35, 2, stride=1, padding=1, bias=False)
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = v1 > 0
        v3 = v1 * -10.398
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x3 = torch.randn(1, 251, 126, 68)
