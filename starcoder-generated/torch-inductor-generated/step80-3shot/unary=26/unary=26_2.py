
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(19, 25, 4, stride=1, padding=0, groups=2)
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * -1.07
        x6 = torch.where(x4, x3, x5)
        return torch.mean(x6)
# Inputs to the model
x2 = torch.randn(15, 19, 11)
