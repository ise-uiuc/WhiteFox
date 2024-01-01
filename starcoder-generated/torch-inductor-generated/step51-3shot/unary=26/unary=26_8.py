
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 1, 9, stride=1, padding=0, bias=False)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        u2 = x1 > 0
        u3 = x1 * -0.208927
        u4 = torch.where(u2, x1, u3)
        x5 = torch.neg(u4)
        x6 = torch.nn.functional.relu6(x5)
        return x6
# Inputs to the model
x3 = torch.randn(2, 7, 14, 15)
