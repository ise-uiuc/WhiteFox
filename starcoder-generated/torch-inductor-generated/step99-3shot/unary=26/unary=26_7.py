
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 8, 5, stride=2, bias=False)
    def forward(self, x):
        e1 = self.conv_t(x)
        e2 = e1 > 0
        e3 = e1 * 18.13
        e4 = torch.where(e2, e1, e3)
        return e4
# Inputs to the model
x = torch.randn(3, 8, 19, 20)
