
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(2, 4, kernel_size=4, stride=5,
                                               padding=6)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.5
max = 1.8
# Inputs to the model
x1 = torch.randn(1, 2, 22)
