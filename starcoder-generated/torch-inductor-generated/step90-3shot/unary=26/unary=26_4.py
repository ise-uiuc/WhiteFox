
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1182, 1182, 3)
    def forward(self, x16):
        x1 = self.conv_t(x16)
        x2 = x1 > 0
        x3 = x1 * 0.044
        x4 = torch.where(x2, x1, x3)
        return x4.neg()
# Inputs to the model
x16 = torch.randn(31, 1182, 151, 6)
