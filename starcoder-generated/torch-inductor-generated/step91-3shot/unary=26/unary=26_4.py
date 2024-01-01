
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(130, 50, 1, stride=1, padding=0, bias=False)[0]
    def forward(self, x36):
        v1 = self.conv_t(x36)
        v2 = v1 > -5.659916
        v3 = v1 * -5.659916
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x36 = torch.randn(2, 130, 14, 13)
