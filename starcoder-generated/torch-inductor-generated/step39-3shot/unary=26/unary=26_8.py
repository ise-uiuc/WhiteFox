
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 3, stride=3, padding=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = torch.nn.LayerNorm([64, 12, 9], 33.313_472_227_885_56)
        return x2(x1)
# Inputs to the model
x = torch.randn(4, 19, 93, 85)
