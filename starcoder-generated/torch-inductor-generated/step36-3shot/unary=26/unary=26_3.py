
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.conv_t = torch.nn.ConvTranspose2d(5, 46, 9, stride=7, padding=7, bias=False)
    def forward(self, x13):
        v1 = self.conv_t(x13)
        v2 = v1 > 0
        v3 = v1 * -0.38
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x13 = torch.randn(4, 5, 83, 28)
