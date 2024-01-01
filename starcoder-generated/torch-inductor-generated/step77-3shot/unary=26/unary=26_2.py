
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(997, 1, 2, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v26 = self.conv_t(x1)
        v27 = v26 > 0
        v28 = v26 * -2.66
        v29 = torch.where(v27, v26, v28)
        return v29
# Inputs to the model
x1 = torch.randn(9, 997, 90, 33)
