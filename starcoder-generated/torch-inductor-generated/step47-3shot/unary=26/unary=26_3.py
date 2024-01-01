
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(443, 438, 6, stride=1, padding=1, bias=False)
    def forward(self, x3):
        b1 = self.conv_t(x3)
        b2 = b1 > 0
        b3 = b1 * 0.9964
        b4 = torch.where(b2, b1, b3)
        return torch.min(torch.sinh(b4), torch.nn.functional.leaky_relu(torch.max(torch.sqrt(b3), torch.sinh(b3)), negative_slope=3.5842))
# Inputs to the model
x3 = torch.randn(43, 443, 43)
