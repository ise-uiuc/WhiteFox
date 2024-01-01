
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 256, 2, padding=13, stride=1, bias=False)
    def forward(self, x):
        v2 = self.conv_t(x)
        f1 = v2 > 0
        f2 = v2 * 0.0
        f3 = torch.where(f1, v2, f2)
        return f3
# Inputs to the model
x = torch.randn(5, 3, 244, 244)
