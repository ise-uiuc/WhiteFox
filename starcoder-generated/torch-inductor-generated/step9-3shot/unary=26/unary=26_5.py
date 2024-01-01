
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 32, 8, stride=1, padding=1)
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v2 = v1 > 0
        v3 = v1 * 0.409
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x0 = torch.randn(7, 8, 7, 7)
