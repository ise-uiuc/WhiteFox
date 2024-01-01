
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(34, 23, 2, stride=1, padding=0, bias=False)
    def forward(self, x4):
        f1 = self.conv_t(x4)
        f2 = f1 > 0
        f3 = f1 * 0.34
        f4 = torch.where(f2, f1, f3)
        return f4
# Inputs to the model
x4 = torch.randn(38, 34, 92, 55)
