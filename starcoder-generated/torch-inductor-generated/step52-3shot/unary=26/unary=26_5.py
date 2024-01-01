
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(76, 42, 11, stride=1, padding=0, bias=False)
    def forward(self, x7):
        o1 = self.conv_t(x7)
        o2 = o1 > 0
        o3 = o1 * -0.4
        o4 = torch.where(o2, o1, o3)
        return o4
# Inputs to the model
x7 = torch.randn(655, 76, 13, 40)
