
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.mask_conv = torch.nn.Conv2d(1, 1, 1)
        torch.nn.init.ones_(self.mask_conv.weight)
    def forward(self, x, y):
        v1 = self.conv(y)
        v2 = self.mask_conv(x)
        v3 = v2 > 0.5
        v4 = torch.where(v3, v1, -v1)
        return v4
# Inputs to the model
x1 = torch.zeros(1, 1, 16, 16)
x2 = torch.ones(1, 1, 16, 16)
