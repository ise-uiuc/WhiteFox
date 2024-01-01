
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 1, 5)
    def forward(self, x4):
        a1 = self.conv_t(x4)
        a2 = a1 > 0.3368896211966404
        a3 = a1 * 3.299655723859835
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x4 = torch.randn(2, 5, 63, 71)
