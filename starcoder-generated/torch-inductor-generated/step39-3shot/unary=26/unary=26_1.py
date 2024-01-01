
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 8, 1, stride=1, padding=0, bias=False)
    def forward(self, x5):
        x1 = self.conv_t(x5)
        x2 = x1 > 0
        x3 = x1 * 1.715
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x5 = torch.randn(2, 10, 13, 18)
