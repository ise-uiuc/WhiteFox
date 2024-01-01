
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=2, bias=False)
    def forward(self, x):
        x6 = self.conv_t(x)
        x7 = x6 > 0
        x8 = x6 * -4.94
        x9 = torch.where(x7, x6, x8)
        return x9
# Inputs to the model
x = torch.randn(2, 4, 35, 42)
