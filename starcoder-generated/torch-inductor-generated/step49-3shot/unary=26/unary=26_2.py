
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 18, 7, stride=5, padding=1, bias=True)
    def forward(self, x):
        h1 = self.conv_t(x)
        h2 = h1 > 0
        h3 = h1 * -0.08
        h4 = torch.where(h2, h1, h3)
        return h4
# Inputs to the model
x = torch.randn(3, 1, 27, 80)
