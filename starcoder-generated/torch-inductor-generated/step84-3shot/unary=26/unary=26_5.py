
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1693, 444, 5, stride=1, padding=1, bias=False)
    def forward(self, x34):
        x2 = self.conv_t(x34)
        v1 = x2 > 0
        v2 = x2 * -9.104
        v3 = torch.where(v1, x2, v2)
        v4 = torch.nn.functional.hardsigmoid(v3)
        return v4
# Inputs to the model
x34 = torch.randn(4, 1693, 30, 21)
