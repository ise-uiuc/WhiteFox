
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(141, 40, 4, stride=2, padding=3, bias=False)
    def forward(self, x12):
        y1 = self.conv_t(x12)
        y2 = y1 > 0
        y3 = y1 * 1.244
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.upsample(y4, size=(174, 110), mode='bicubic')
# Inputs to the model
x12 = torch.randn(16, 141, 77, 60)
