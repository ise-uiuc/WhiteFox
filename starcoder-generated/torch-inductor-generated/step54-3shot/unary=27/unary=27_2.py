
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 3, 12, bias=False)
        self.conv = torch.nn.Conv2d(4, 4, 10, stride=20, bias=True, padding_mode='zeros', padding=0)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.conv_t(x1, x2.size())
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 1
max = 6
# Inputs to the model
x1 = torch.randn(1, 8, 32, 19)
x2 = torch.randn(1, 16)
