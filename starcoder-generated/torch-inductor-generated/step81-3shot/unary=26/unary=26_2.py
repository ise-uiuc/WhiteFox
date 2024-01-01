
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(32, 128, 4, stride=2, padding=1)
    def forward(self, y):
        w1 = self.conv_t(y)
        w2 = w1 > 0
        w3 = w1 * -0.51027
        w4 = torch.where(w2, w1, w3)
        return w4
# Inputs to the model
y = torch.randn(8, 32, 10, 30)
