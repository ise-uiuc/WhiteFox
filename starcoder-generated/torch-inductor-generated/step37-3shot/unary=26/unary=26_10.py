
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(258, 282, 2, padding=0, bias=False)
    def forward(self, x3):
        a1 = self.conv_t(x3)
        a2 = a1 > 0
        a3 = a1 * -0.390
        a4 = torch.where(a2, a1, a3)
        return torch.nn.functional.hardtanh(torch.nn.functional.relu(a4))
# Inputs to the model
x3 = torch.randn(48, 258, 83, 27)
