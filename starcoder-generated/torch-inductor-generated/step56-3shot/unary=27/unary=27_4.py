
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 3, kernel_size=(1, 1, 19), stride=(9, 5, 1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.8
max = 0.9
# Inputs to the model
x1 = torch.randn(1, 1, 60, 89, 89)
y = torch.randn(1, 3, 60, 89, 89)
