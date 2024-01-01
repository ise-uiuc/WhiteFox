
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 256//16, 1, stride=1, padding=0, groups=16)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2, 3)
        v2 = self.conv(v1)
        v3 = v2.permute(1, 0, 2, 3)
        return v3
# Inputs to the model
x1 = torch.randn(256, 9, 9)
