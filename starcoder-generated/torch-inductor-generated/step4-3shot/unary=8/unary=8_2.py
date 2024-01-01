
class Model(torch.nn.Module):
    def __init__(self):
         super().__init__()
         m1 = torch.nn.AvgPool2d(3, stride=2, padding=2, count_include_pad=True)
         m2 = torch.nn.Conv2d(8, 26, 5, stride=2, padding=0, groups=4, dilation=2)
         m3 = torch.nn.ConvTranspose2d(26, 6, 4, stride=2, padding=2, output_padding=2, groups=1, dilation=1)
         self.m123 = torch.nn.Sequential(m1, m2, m3)
    def forward(self, x1):
        v1 = self.m123(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
