
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 6, 5, padding=2, stride=1)
        self.negative_slope = 0.2
    def forward(self, x3):
        m1 = self.conv_t(x3)
        m2 = m1 > 0
        m3 = m1 * self.negative_slope
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.dropout(m4, p=0.3850582905077984, train=False)
# Inputs to the model
x3 = torch.randn(5, 4, 83, 76)
