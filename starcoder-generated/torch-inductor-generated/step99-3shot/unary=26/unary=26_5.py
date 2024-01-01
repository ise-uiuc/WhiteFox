
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(67, 1, 2, stride=1, padding=1, bias=False)
    def forward(self, x):
        a1 = self.conv_t(x)
        a2 = a1 > 0
        a3 = a1 * 0.056
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x = torch.randn(1, 67, 82, 33)
