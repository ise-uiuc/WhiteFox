
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(333, 444, 9, stride=1, padding=4, bias=False)
    def forward(self, x12):
        b1 = self.conv_t(x12)
        b2 = b1 > 0
        b3 = b1 * -0.1
        b4 = torch.where(b2, b1, b3)
        return torch.nn.functional.interpolate(torch.nn.ReLU()(b4), (296, 287))
# Inputs to the model
x12 = torch.randn(2, 333, 172)
