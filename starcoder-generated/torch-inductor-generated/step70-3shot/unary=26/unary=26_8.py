
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 1, 2, 9, 8)
    def forward(self, x5):
        s1 = self.conv_t(x5)
        s2 = s1 > 0
        s3 = s1 * -13.80
        s4 = torch.where(s2, s1, s3)
        return s4
# Inputs to the model
x5 = torch.randn(1, 1, 27, 9)
