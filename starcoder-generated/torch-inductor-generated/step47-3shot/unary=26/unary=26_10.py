
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(139, 64, 3, stride=2, padding=0, bias=True)
    def forward(self, x31):
        a1 = self.conv_t(x31)
        a4 = torch.tanh(a1) ** 2
        return a4
# Inputs to the model
x31 = torch.randn(6, 139, 9, 9)
