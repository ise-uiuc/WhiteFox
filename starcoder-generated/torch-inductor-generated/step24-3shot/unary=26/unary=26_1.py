
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 49, 3, stride=1)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        m1 = x2 > 0
        v2 = v1 * torch.where(m1, x2, m1)
        m2 = v2 < 5
        v5 = v2 * m2
        return v5
# Inputs to the model
x2 = torch.randn(8, 3, 8, 8)
