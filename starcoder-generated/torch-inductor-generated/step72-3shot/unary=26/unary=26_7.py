
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(305, 3, 6, stride=2, padding=4, output_padding=3)
    def forward(self, x2):
        n7 = self.conv_t(x2)
        n8 = n7 > 0
        n9 = n7 * -0.8254051542853583
        n10 = torch.where(n8, n7, n9)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.hardtanh(n10, -8, 8), (1, 1))
# Inputs to the model
x2 = torch.randn(56, 305, 26, 26)
