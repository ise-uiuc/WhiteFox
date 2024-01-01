
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(72, 160, 5, 2, 3, 1)
    def forward(self, x1):
        y1 = self.conv_t(x1)
        y2 = y1 > 0
        y3 = y1 * 0.125
        y4 = torch.where(y2, y1, y3)
        y5 = torch.nn.functional.relu(y4)
        return y5
# Inputs to the model
x1 = torch.randn(77, 72, 16, 16)
