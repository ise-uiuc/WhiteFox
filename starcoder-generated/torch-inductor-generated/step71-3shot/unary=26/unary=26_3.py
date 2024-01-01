
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(488, 120, 19, stride=3, padding=6, bias=False)
    def forward(self, x1):
        w1 = self.conv_t(x1)
        w2 = w1 > 0
        w3 = w1 * 2.669
        w4 = torch.where(w2, w1, w3)
        return torch.nn.functional.hardtanh(w4, -2.025, 2.025)
# Inputs to the model
x1 = torch.randn(3, 488, 44, 43)
