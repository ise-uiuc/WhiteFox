
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=1, mode='bilinear')
        self.dconv = torch.nn.ConvTranspose2d(50, 1, 1, stride=1, padding=1)
    def forward(self, x3):
        v1 = self.upsample(x3)
        v2 = self.dconv(v1)
        return v2
# Inputs to the model
x3 = torch.randn(1, 50, 10, 10)
