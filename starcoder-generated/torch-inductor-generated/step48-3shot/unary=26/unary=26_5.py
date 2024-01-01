
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 26, 4, stride=2, padding=1)
    def forward(self, x7):
        v1 = self.conv_t(x7)
        v2 = v1 > 0
        v3 = v1 * -0.5
        v4 = torch.where(v2, v1, v3)
        v5 = v4 > 1.1
        v6 = v4 * 1.26
        return torch.exp(torch.where(v5, v4, v6))
# Inputs to the model
x7 = torch.randn(5, 15, 12, 15)
