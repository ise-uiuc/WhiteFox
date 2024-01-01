
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1, bias=False)
    def forward(self, x4):
        w = torch.randn(2, 3, 3, 3)
        z = torch.nn.functional.conv2d(x4, w, bias=None, stride=2, padding=0)
        z1 = self.conv_t(x4)
        return torch.nn.functional.interpolate(z1, scale_factor=[1.0, 1.0])
# Inputs to the model
x4 = torch.randn(3, 3, 15, 15)
