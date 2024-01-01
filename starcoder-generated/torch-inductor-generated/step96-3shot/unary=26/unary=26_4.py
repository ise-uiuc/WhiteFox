
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 112, 5, stride=2, padding=0)
    def forward(self, x):
        x23 = self.conv_t(x)
        x24 = x23 > 0
        x25 = x23 * 0
        x26 = torch.where(x24, x23, x25)
        x27 = torch.nn.functional.softmax(x23, dim=1)
        return torch.nn.functional.interpolate(x23, size=[51, 96], mode='size')
# Inputs to the model
x = torch.randn(1, 35, 108, 176)
