
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(314, 164, 3, stride=1, padding=2)
    def forward(self, x4):
        f1 = self.conv_t(x4)
        f2 = f1 > 0
        f3 = f1 * -0.799
        f4 = torch.where(f2, f1, f3)
        f5 = f4.max(dim=1).values
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.hardtanh(f5, -8, 8), (1, 1))
# Inputs to the model
x4 = torch.randn(5, 314, 5, 21)
