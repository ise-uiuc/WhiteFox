
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 180, 3, stride=1, padding=4, bias=False)
        self.conv_t = torch.nn.ConvTranspose2d(180, 144, 6, stride=1, padding=4, bias=False)
    def forward(self, x2):
        h1 = torch.nn.functional.interpolate(x2, scale_factor=[0.373, 1.0])
        h2 = self.conv(h1)
        h3 = h2 > 0
        h4 = h2 * -0.898
        h5 = torch.where(h3, h2, h4)
        h6 = self.conv_t(h5)
        return torch.nn.functional.interpolate(torch.nn.Softplus()(h6), scale_factor=[1.0, 2.0])
# Input to the model
x2 = torch.randn(1, 1, 71, 9)
