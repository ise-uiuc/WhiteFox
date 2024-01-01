
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 1024, kernel_size=3, dilation=3, stride=1, padding=0)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
# Inputs to the model
x1 = torch.randn(1, 1024, 64, 64)
