
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, bias=False)
    def forward(self, x4):
        x1 = self.conv_t(x4)
        x2 = x1 > 0
        x3 = x1 * 9.89329
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.leaky_relu(x4)
# Inputs to the model
x4 = torch.randn(2, 16, 125)
