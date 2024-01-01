
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose3d(14, 41, 2, stride=1, padding=0, bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(330, 427, 4, stride=2, padding=1, bias=False)
    def forward(self, x16):
        x1 = self.conv_t(x16)
        x2 = x1 > 0
        x3 = x1 * -0.08
        x4 = torch.where(x2, x1, x3)
        return self.conv_t2(x4)
# Inputs to the model
x16 = torch.randn(2, 14, 49, 84, 51)
