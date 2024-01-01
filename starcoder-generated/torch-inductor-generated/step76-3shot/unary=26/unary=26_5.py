
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(43, 159, 6, stride=2, padding=0, output_padding=1, groups=2, bias=False)
    def forward(self, x8) -> torch.Tensor:
        w1 = self.conv_t(x8)
        w2 = w1 > 0
        w3 = w1 * -0.119
        w4 = torch.where(w2, w1, w3)
        return w1
# Inputs to the model
x8 = torch.randn(1, 43, 6, 4)
