
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(248, 3, 3, stride=2, padding=2, output_padding=1)
    def forward(self, x2):
        n1 = self.conv_t(x2)
        n2 = n1 > 0
        n3 = n1 * -0.5
        n4 = torch.where(n2, n1, n3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.hardtanh(n4, -8, 8), (1, 1))
# Inputs to the model
x2 = torch.randn(89, 248, 7, 25)
