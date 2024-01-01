
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 96, 7, stride=1, padding=0, bias=False)
    def forward(self, x6):
        v1 = self.conv_t(x6)
        v2 = v1 > 0
        v4 = torch.where(v2, v1, v1)
        v3 = -0.9964
        return v4 * v3
# Inputs to the model
x6 = torch.randn(14, 5, 3, 14)
