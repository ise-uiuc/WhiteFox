
class Model(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(42, 2, 2, stride=stride)
    def forward(self, x30):
        x19 = self.conv_t(x30)
        x20 = x19 > 0
        x21 = x19 * -0.286
        x22 = torch.where(x20, x19, x21)
        return torch.nn.functional.hardtanh(x22)
stride = 2
# Inputs to the model
x30 = torch.randn(7, 42, 13, 24)
