
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 16, (16, 48), stride=2, bias=True)
    def forward(self, x12):
        f1 = self.conv_t(x12)
        f2 = f1 > 0
        f3 = f1 * 0.13
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.flatten(f4, 1)
# Inputs to the model
x12 = torch.randn(8, 64, 9, 38)
