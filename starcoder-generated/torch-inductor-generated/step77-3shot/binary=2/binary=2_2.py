
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=3, groups=2, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 3.0
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
