
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(512, 256, 2, stride=1, padding=2, bias=True)
        self.conv_t.weight.data = torch.eye(256, 512).double()
    def forward(self, x8, x9, x10, x11):
        y10 = self.conv_t(x8, x9, x10, x11)
        y11 = y10 > -4232
        y12 = y10 * 0.8900
        y13 = torch.where(y11, y10, y12)
        return y13
# Inputs to the model
from torch.nn.functional import upsample_nearest
conv_w = torch.nn.Conv2d(512, 256, 2, stride=1, padding=2, bias=True)
conv_w.weight.data = torch.eye(256, 512).double()
x8 = torch.randn(2, 512, 4, 8)
x9 = upsample_nearest(x8, scale_factor=[0.5, 1.0])
x10 = upsample_nearest(x8, scale_factor=[0.5, 1.0])
x11 = upsample_nearest(x8, scale_factor=[0.5, 1.0])
