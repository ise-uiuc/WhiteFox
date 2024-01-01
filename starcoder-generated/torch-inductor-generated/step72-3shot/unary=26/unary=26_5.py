
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(93, 6, 5, stride=2, padding=4)
    def forward(self, x1):
        c1 = self.conv_t(x1)
        c2 = c1 > 0.0
        c3 = c1 * 0.006321958026499228
        c4 = torch.where(c2, c1, c3)
        c5 = c4.max(dim=3).values
        c6 = c5.mean(dim=1)
        return torch.nn.functional.hardsigmoid(c6, -8, 8)
# Inputs to the model
x1 = torch.randn(5, 93, 31, 9)
