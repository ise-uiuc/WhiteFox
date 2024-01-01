
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 5, 1, stride=1, padding=1)
    def forward(self, x1, other=0, padding1=1, padding2=1, groups=0):
        v1 = self.conv(x1)
        if other == 0:
            other = torch.randn(v1.shape)
        if padding1 == 1:
            x1 = F.pad(x1, (padding2, padding2, padding2, padding2), "constant", 0)
        v2 = x1 + other
        if groups == 0:
            groups = 16
        v3 = F.conv2d(v2, torch.randn(16,5,3,3), stride=1, padding=1, groups=groups, dilation=2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 56, 56)
