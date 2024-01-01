
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 7, 5, stride=1, padding=2, bias=True)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * -1.5279
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x3 = torch.randn(3, 5, 7, 9)
