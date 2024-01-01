
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(85, 198, 5, stride=1, padding=0, bias=False)
    def forward(self, x9):
        w1 = self.conv_t(x9)
        w2 = w1 > 0
        w3 = w1 * -0.745
        w4 = torch.where(w2, w1, w3)
        return w4
# Inputs to the model
x9 = torch.randn(65, 85, 28, 75)
