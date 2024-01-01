
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 2, stride=2)
    def forward(self, x):
        y1 = self.conv_t(x.reshape(1, 4, 8, 8))
        y2 = y1 > 0
        y3 = y1 * 0.02
        y4 = torch.where(y2, y1, y3)
        return y4.detach()
# Inputs to the model
x = torch.randn(22, 4)
