
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1, padding=None, x2=None, v1=1, v2=None, weight=2, ksize=3, groups=1):
        if padding == None:
            padding = torch.randn(weight.shape)
        x1 = self.conv(x1)
        if x2 == None:
            x2 = x1 + padding
        v2 = torch.nn.functional.conv2d(v1, weight, padding=ksize // 2, groups=groups)
        v2 = v2 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 35, 35)
