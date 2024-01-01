
class Model(torch.nn.Module):
    def __init__(self, min=0.1, max=0.3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, groups=3, out_channels=3, kernel_size=(2, 3), stride=(1, 2), padding=(1, 1), bias=True, dilation=1)
        self.a = min
        self.b = max
    def forward(self, x0):
        v0 = self.conv(x0)
        v1 = torch.clamp_min(v0, self.a)
        v2 = torch.clamp_max(v1, self.b)
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 26, 26)

model = Model(0.6, 0.02)
