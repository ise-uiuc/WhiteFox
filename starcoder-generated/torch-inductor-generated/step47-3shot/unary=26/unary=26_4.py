
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(52, 16, 8, stride=4, padding=2, bias=True)
    def forward(self, x22):
        z1 = self.conv_t(x22)
        z2 = z1 > 0
        z3 = z1 * 0.353
        z4 = torch.where(z2, z3, z1)
        return torch.max(z4, 0.546)
# Inputs to the model
x22 = torch.randn(33, 52, 8)
