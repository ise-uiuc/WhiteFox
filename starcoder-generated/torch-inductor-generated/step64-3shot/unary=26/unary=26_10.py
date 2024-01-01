
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 3, 5, stride=1, padding=2, bias=False)
    def forward(self, x3):
        s1 = self.conv_t(x3)
        s2 = s1 > 0
        s3 = s1 * 0.081
        s4 = torch.where(s2, s1, s3)
        return torch.abs(s4)
# Inputs to the model
x3 = torch.randn(42, 5, 32, 13)
