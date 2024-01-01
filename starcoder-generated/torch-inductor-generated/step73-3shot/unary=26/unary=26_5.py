
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(5, 3, 8, stride=1, padding=0, bias=True)
    def forward(self, x24):
        f1 = self.conv_t(x24)
        f2 = f1 > 1
        f3 = f1 * 0.771
        f4 = torch.where(f2, f1, f3)
        return f4
# Inputs to the model
x24 = torch.randn(1, 5, 12, 24, 75, dtype=torch.float32)
