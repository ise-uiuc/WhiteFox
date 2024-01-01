
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 5, 1, stride=(1, 2, 2), padding=1)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        return v2
min = 5.257919054398539
# Inputs to the model
x1 = torch.randn(1, 3, 17, 7, 9)
