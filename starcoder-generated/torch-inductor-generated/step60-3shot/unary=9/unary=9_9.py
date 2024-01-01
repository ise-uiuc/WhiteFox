
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 12, kernel_size=9, stride=4, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1_add = v1 + 3
        v1_clamp_min = v1_add.clamp(0, 6)
        v1_clamp_max = v1_clamp_min.clamp(min=0, max=6)
        v1_div = v1_clamp_max / 6
        return v1_div
# Inputs to the model
x1 = torch.randn(7, 3, 128, 128)
