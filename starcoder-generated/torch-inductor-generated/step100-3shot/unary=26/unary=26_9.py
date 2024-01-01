
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(17, 98, 22, stride=1, padding=1, bias=False)
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v1 = torch.floor(v1)
        v2 = v1 > 0
        v3 = v1 * 0.145
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x0 = torch.randn(3, 17, 5, 22)
