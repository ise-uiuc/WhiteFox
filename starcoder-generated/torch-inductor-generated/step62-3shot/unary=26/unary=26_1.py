
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 2, 1)
    def forward(self, x2):
        h1 = self.conv_t(x2)
        h2 = torch.nn.functional.interpolate(h1, (6, 9))
        h3 = torch.nn.functional.interpolate(h2, (12, 16))
        h4 = torch.nn.functional.interpolate(h3, (24, 30))
        return h4
# Inputs to the model
x2 = torch.randn(1, 3, 16, 24)
