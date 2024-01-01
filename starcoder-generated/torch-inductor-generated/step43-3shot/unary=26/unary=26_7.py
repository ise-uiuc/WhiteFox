
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 8, 5, stride=1, padding=5, bias=False)
    def forward(self, x1):
        y1 = self.conv_t(x1)
        y2 = y1 > 0
        y3 = y1 * 5.893
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x1 = torch.randn(1, 4, 1, 7, requires_grad=False)
