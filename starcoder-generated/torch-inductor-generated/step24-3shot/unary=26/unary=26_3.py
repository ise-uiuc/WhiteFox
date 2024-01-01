
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 7, 8, stride=8)
    def forward(self, x3):
        v3 = self.conv_t(x3)
        return v3
# Inputs to the model
x3 = torch.randn(3, 1, 56, 56)
