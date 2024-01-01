
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 4, 1, stride=1, padding=0)
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -0.620
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.interpolate(v4, size=4)
# Inputs to the model
x4 = torch.randn(5, 2, 20, 18)
