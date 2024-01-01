
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(20, 7, 7, stride=2, bias=False, padding=2)
    def forward(self, x9):
        x1 = self.conv_t(x9)
        u2 = x1 > 0
        u3 = x1 * -0.627582
        u4 = torch.where(u2, x1, u3)
        x5 = torch.neg(u4)
        x6 = torch.nn.functional.relu6(x5)
        x7 = torch.abs(x6)
        x8 = torch.floor(x7)
        return x8
# Inputs to the model
x9 = torch.randn(1, 20, 13, 6)
