
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 512, 4, stride=2, padding=1, bias=False)
    def forward(self, x16, x17):
        o1 = self.conv_t(x16)
        o2 = o1 + x17
        o3 = o2 > 0
        o4 = torch.where(o3, o1, o2)
        return o4
# Inputs to the model
x16 = torch.randn(3, 256, 26, 26)
x17 = torch.randn(3, 512, 7, 7)
