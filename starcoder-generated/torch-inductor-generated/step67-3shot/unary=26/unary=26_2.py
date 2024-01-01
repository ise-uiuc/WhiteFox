
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convt2d = torch.nn.ConvTranspose2d(1, 21, 5, 2)
    def forward(self, x11):
        x12 = self.convt2d(x11)
        x13 = x12 > 0
        x14 = x12 * -0.64
        x15 = torch.where(x13, x12, x14)
        return x15
# Inputs to the model
x11 = torch.randn(9, 1, 12, 23)
